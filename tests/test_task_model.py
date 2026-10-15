"""Tests for task lifecycle, updates, completion, expiry, and replacement order."""

import numpy as np
import pytest
from semantic_noma.config import TaskConfig
from semantic_noma.task_model import TaskManager


def test_task_initialization():
    """Verify task manager initializes fresh tasks for all users within bounds."""
    config = TaskConfig(demand_range_bits=(3000, 5000), deadline_range_frames=(3, 5), importance_range=(0.4, 0.8))
    tm = TaskManager(num_users=8, config=config, seed=42)
    tm.reset()

    assert len(tm.active_tasks) == 8
    for u in range(8):
        task = tm.active_tasks[u]
        assert 3000 <= task.original_demand_bits <= 5000
        assert task.remaining_demand_bits == task.original_demand_bits
        assert 3 <= task.initial_deadline_frames <= 5
        assert task.remaining_deadline_frames == task.initial_deadline_frames
        assert 0.4 <= task.importance_weight <= 0.8
        assert not task.completed
        assert not task.missed


def test_task_completion_and_service_capping():
    """Verify service delivers bits up to remaining demand and marks completion."""
    config = TaskConfig(demand_range_bits=(5000, 5000), deadline_range_frames=(5, 5))
    tm = TaskManager(num_users=2, config=config, seed=1)
    tm.reset()

    # User 0 gets 2000 bits (partial), User 1 gets 6000 bits (overservice)
    service = np.array([2000.0, 6000.0])
    useful_bits, progress, mean_u, completed, expired = tm.process_frame(service, current_frame=0)

    # User 0
    assert useful_bits[0] == 2000.0
    # Remaining was 5000 - 2000 = 3000
    assert len(completed) == 1
    assert completed[0].user_id == 1  # User 1 completed

    # User 1 delivered bits capped at original demand 5000
    assert useful_bits[1] == 5000.0

    # User 1 replacement task was generated for next frame
    assert tm.active_tasks[1].arrival_frame == 1
    assert tm.active_tasks[1].remaining_demand_bits == tm.active_tasks[1].original_demand_bits
    # New task received no service in frame 0
    assert len(tm.active_tasks[1].service_history) == 0


def test_task_deadline_expiry_and_penalty_fraction():
    """Verify task expiry when deadline reaches zero and no completion counted on miss."""
    config = TaskConfig(demand_range_bits=(10000, 10000), deadline_range_frames=(2, 2))
    tm = TaskManager(num_users=1, config=config, seed=1)
    tm.reset()

    # Frame 0: give 2000 bits (remaining 8000, deadline decrements from 2 to 1)
    _, _, _, completed_0, expired_0 = tm.process_frame(np.array([2000.0]), current_frame=0)
    assert len(completed_0) == 0
    assert len(expired_0) == 0

    # Frame 1: give 1000 bits (remaining 7000, deadline decrements from 1 to 0 -> expires!)
    _, _, _, completed_1, expired_1 = tm.process_frame(np.array([1000.0]), current_frame=1)
    assert len(completed_1) == 0
    assert len(expired_1) == 1

    expired_task, remaining_fraction = expired_1[0]
    assert expired_task.missed
    # 7000 remaining out of 10000 = 0.7
    assert pytest.approx(0.7) == remaining_fraction


def test_completion_on_deadline_frame():
    """Verify that completing a task on its final deadline frame does not count as missed."""
    config = TaskConfig(demand_range_bits=(4000, 4000), deadline_range_frames=(1, 1))
    tm = TaskManager(num_users=1, config=config, seed=1)
    tm.reset()

    # Task has deadline = 1 frame. If we deliver 4000 bits, it completes and is NOT expired/missed.
    _, _, _, completed, expired = tm.process_frame(np.array([4000.0]), current_frame=0)
    assert len(completed) == 1
    assert len(expired) == 0
    assert not completed[0].missed
