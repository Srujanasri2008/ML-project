"""Task representation and queue lifecycle for semantic communication."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .config import TaskConfig


@dataclass
class SemanticTask:
    task_id: int
    user_id: int
    importance_weight: float
    original_demand_bits: float
    remaining_demand_bits: float
    initial_deadline_frames: int
    remaining_deadline_frames: int
    arrival_frame: int
    completed: bool = False
    missed: bool = False
    completion_frame: Optional[int] = None
    service_history: List[float] = field(default_factory=list)

    @property
    def remaining_demand_ratio(self) -> float:
        if self.original_demand_bits <= 0:
            return 0.0
        return float(np.clip(self.remaining_demand_bits / self.original_demand_bits, 0.0, 1.0))

    @property
    def remaining_deadline_ratio(self) -> float:
        if self.initial_deadline_frames <= 0:
            return 0.0
        return float(np.clip(self.remaining_deadline_frames / self.initial_deadline_frames, 0.0, 1.0))


class TaskManager:
    """Manages active tasks, service updates, completions, expiries, and replacements."""

    def __init__(self, num_users: int, config: TaskConfig, seed: Optional[int] = 42):
        self.num_users = num_users
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.task_counter = 0
        self.active_tasks: Dict[int, SemanticTask] = {}
        self.completed_tasks: List[SemanticTask] = []
        self.expired_tasks: List[SemanticTask] = []

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset task manager and initialize fresh tasks for all users at frame 0."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.task_counter = 0
        self.active_tasks.clear()
        self.completed_tasks.clear()
        self.expired_tasks.clear()

        for user_id in range(self.num_users):
            self.active_tasks[user_id] = self._create_task(user_id=user_id, arrival_frame=0)

    def _create_task(self, user_id: int, arrival_frame: int) -> SemanticTask:
        """Create a new synthetic task according to configured distributions."""
        self.task_counter += 1
        d_min, d_max = self.config.demand_range_bits
        dl_min, dl_max = self.config.deadline_range_frames
        w_min, w_max = self.config.importance_range

        # Sample demand, deadline, and importance weight
        demand = float(self.rng.integers(d_min, d_max + 1))
        deadline = int(self.rng.integers(dl_min, dl_max + 1))
        importance = float(self.rng.uniform(w_min, w_max))

        return SemanticTask(
            task_id=self.task_counter,
            user_id=user_id,
            importance_weight=importance,
            original_demand_bits=demand,
            remaining_demand_bits=demand,
            initial_deadline_frames=deadline,
            remaining_deadline_frames=deadline,
            arrival_frame=arrival_frame,
        )

    def process_frame(
        self,
        service_bits: np.ndarray,
        current_frame: int,
    ) -> Tuple[np.ndarray, np.ndarray, float, List[SemanticTask], List[Tuple[SemanticTask, float]]]:
        """Update active tasks with delivered service bits for current frame.
        
        Order of operations:
            1. Compute useful service s_i(t) = min(q_i(t), c_i(t))
            2. Compute normalized progress u_i(t) = w_i * (s_i(t) / D_i)
            3. Compute mean semantic utility U_sem(t) = (1/K) * sum(u_i(t))
            4. Decrement remaining demand q_i(t)
            5. Check completion (q_i(t) <= 1e-9). Completed tasks are recorded.
            6. For incomplete tasks, decrement remaining deadline.
            7. If deadline reaches 0, task expires. Record missed task and remaining demand fraction.
            8. Replace completed/expired tasks with fresh tasks for NEXT frame (no service this frame).
            
        Returns:
            (useful_bits, semantic_utilities, mean_sem_utility, newly_completed, newly_expired_with_penalty_fraction)
        """
        useful_bits = np.zeros(self.num_users, dtype=np.float64)
        progress_per_user = np.zeros(self.num_users, dtype=np.float64)
        newly_completed: List[SemanticTask] = []
        newly_expired: List[Tuple[SemanticTask, float]] = []

        # 1. Deliver service and calculate utility
        for user_id in range(self.num_users):
            task = self.active_tasks[user_id]
            c_i = max(0.0, float(service_bits[user_id]))
            s_i = min(task.remaining_demand_bits, c_i)
            useful_bits[user_id] = s_i
            task.service_history.append(s_i)

            u_i = task.importance_weight * (s_i / task.original_demand_bits)
            progress_per_user[user_id] = u_i

            # 4. Reduce remaining demand
            task.remaining_demand_bits -= s_i

        mean_semantic_utility = float(np.mean(progress_per_user))

        # 5-8. Check completion, deadline expiry, and replacements
        for user_id in range(self.num_users):
            task = self.active_tasks[user_id]

            if task.remaining_demand_bits <= 1e-9:
                # Successfully completed (even if remaining_deadline was 0 or 1!)
                task.completed = True
                task.remaining_demand_bits = 0.0
                task.completion_frame = current_frame
                newly_completed.append(task)
                self.completed_tasks.append(task)
                # Generate replacement task for next frame (receives no service this frame)
                self.active_tasks[user_id] = self._create_task(
                    user_id=user_id, arrival_frame=current_frame + 1
                )
            else:
                # Incomplete: decrement deadline
                task.remaining_deadline_frames -= 1
                if task.remaining_deadline_frames <= 0:
                    # Deadline expired
                    task.missed = True
                    # Missed demand fraction after current frame's service
                    remaining_fraction = task.remaining_demand_bits / task.original_demand_bits
                    newly_expired.append((task, remaining_fraction))
                    self.expired_tasks.append(task)
                    # Generate replacement task for next frame (receives no service this frame)
                    self.active_tasks[user_id] = self._create_task(
                        user_id=user_id, arrival_frame=current_frame + 1
                    )

        return useful_bits, progress_per_user, mean_semantic_utility, newly_completed, newly_expired
