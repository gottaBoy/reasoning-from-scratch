"""Pin the Day 7 teaching traces without claiming GPU timing or KV allocation."""

from collections import Counter

import pytest
import torch

from learning.day07_memory import engine_lesson, small_config, storage_lesson
from learning.mini_transformer import Model
from learning.mini_vllm.engine import Engine, Request
from learning.mini_vllm.scheduler import Job, simulate


def lesson_jobs():
    return [Job("A", 3), Job("B", 1), Job("C", 2, arrival=1)]


@pytest.mark.parametrize("mode,expected", [
    ("sequential", [("A",), ("A",), ("A",), ("B",), ("C",), ("C",)]),
    ("static", [("A", "B"), ("A",), ("A",), ("C",), ("C",)]),
    ("continuous", [("A", "B"), ("A", "C"), ("A", "C")]),
])
def test_day07_documented_traces(mode, expected):
    report = simulate(lesson_jobs(), mode=mode, slots=2, token_budget=2)
    assert [tuple(row["executed"]) for row in report["trace"]] == expected
    assert report["rounds"] == len(expected)


@pytest.mark.parametrize("mode", ["sequential", "static", "continuous"])
@pytest.mark.parametrize("budget", [1, 2])
def test_day07_work_conservation_arrivals_and_counter_conventions(mode, budget):
    jobs = lesson_jobs()
    report = simulate(jobs, mode=mode, slots=2, token_budget=budget)
    counts = Counter(name for row in report["trace"] for name in row["executed"])
    assert counts == {job.name: job.tokens for job in jobs}
    for job in jobs:
        rounds = [row["round"] for row in report["trace"] if job.name in row["executed"]]
        assert rounds[0] >= job.arrival
        assert report["first_round"][job.name] == rounds[0] + 1
        assert report["finish_round"][job.name] == rounds[-1] + 1
    for row in report["trace"]:
        assert len(row["executed"]) <= budget
        assert len(set(row["executed"])) == len(row["executed"])


def test_day07_single_item_budget_does_not_promise_fewer_rounds():
    reports = [simulate(lesson_jobs(), mode=mode, slots=2, token_budget=1)
               for mode in ("sequential", "static", "continuous")]
    assert [report["rounds"] for report in reports] == [6, 6, 6]


@pytest.mark.parametrize("mode", ["sequential", "static", "continuous"])
def test_day07_idle_rounds_do_not_execute_requests_before_arrival(mode):
    report = simulate([Job("late", 1, arrival=2)], mode=mode)
    assert [row["executed"] for row in report["trace"]] == [[], [], ["late"]]
    assert report["first_round"] == report["finish_round"] == {"late": 3}


def test_day07_physical_pool_dimensions_and_page_bytes():
    report = storage_lesson()
    assert report["storage_shape"] == [6, 2, 2, 1, 2, 2]
    assert report["storage_bytes"] == 384
    assert report["bytes_per_block"] == 64
    assert all(event["pool_bytes"] == 384 for event in report["events"])


def test_day07_drawer_directory_distinguishes_reservation_from_valid_length():
    events = {event["event"]: event for event in storage_lesson()["events"]}
    assert events["reserve_A_B"]["requests"]["A"] == {
        "table": [0, 1], "capacity": 4, "lengths": [0, 0],
    }
    assert events["prefill_A3_B2"]["requests"]["A"]["lengths"] == [3, 3]
    assert events["append_A_to4"]["requests"]["A"]["lengths"] == [4, 4]
    assert events["reserve_C_empty"]["requests"]["C"] == {
        "table": [4, 5, 2], "capacity": 6, "lengths": [0, 0],
    }
    assert events["write_C5"]["requests"]["C"]["lengths"] == [5, 5]


def test_day07_reused_noncontiguous_pages_preserve_logical_kv_order():
    report = storage_lesson()
    assert report["C_position"] == {"logical": 4, "physical": 2, "offset": 0}
    assert report["C_layer0_keys_in_logical_order"] == [
        [[[300., 301.], [302., 303.], [304., 305.], [306., 307.], [308., 309.]]],
    ]
    assert report["released_handle_rejected"]
    assert report["release_does_not_zero_page"]


def test_day07_pool_exhaustion_and_release_keep_storage_reusable():
    report = storage_lesson()
    events = {event["event"]: event for event in report["events"]}
    assert events["write_C5"]["free"] == [3]
    assert events["reject_D_without_mutation"]["free"] == [3]
    assert report["oom_without_mutation"]
    assert report["all_blocks_free"] and report["storage_address_unchanged"]
    assert events["release_all"]["requests"] == {}


def test_day07_real_tiny_engine_buckets_and_cache_capacity_align():
    report = engine_lesson()
    assert report["reserved_capacity"] == {"A": 5, "B": 3, "C": 3}
    assert {name: len(ids) for name, ids in report["outputs"].items()} == {
        "A": 3, "B": 1, "C": 2,
    }
    assert report["independent_greedy_alignment"] == "passed"
    assert report["batches"][0] == {
        "round": 0, "batch_size": 2, "cache_position": 0, "input_tokens_each": 3,
    }
    assert report["all_blocks_free"] and report["storage_address_unchanged"]


def test_day07_real_engine_finally_reclaims_pages_after_forward_failure(monkeypatch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = Model(small_config())
    engine = Engine(model, blocks=4, block_size=2)
    address = engine.pool.storage.data_ptr()

    def fail_forward(*args, **kwargs):
        assert "failure" in engine.pool.requests
        assert len(engine.pool.free) == 2
        raise RuntimeError("lesson failure")

    monkeypatch.setattr(model, "forward", fail_forward)
    with pytest.raises(RuntimeError, match="lesson failure"):
        engine.run([Request("failure", [1, 2], 2)])
    assert not engine.pool.requests and len(engine.pool.free) == 4
    assert len(set(engine.pool.free)) == 4
    assert engine.pool.storage.data_ptr() == address
