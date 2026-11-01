"""Discrete rounds, not wall-clock/GPU throughput measurements."""

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    name: str
    tokens: int
    arrival: int = 0


def simulate(jobs, mode="continuous", slots=2, token_budget=2):
    if mode not in ("sequential", "static", "continuous"):
        raise ValueError("Invalid scheduling mode.")
    if min(slots, token_budget) < 1 or len({j.name for j in jobs}) != len(jobs):
        raise ValueError("Positive limits and unique ids required.")
    if any(j.tokens < 1 or j.arrival < 0 for j in jobs):
        raise ValueError("Jobs need positive tokens and nonnegative arrivals.")
    pending = deque(sorted(jobs, key=lambda j: j.arrival))
    waiting, running = deque(), deque()
    remaining = {j.name: j.tokens for j in jobs}
    first, finished, trace = {}, {}, []
    time = 0
    capacity = 1 if mode == "sequential" else slots
    while pending or waiting or running:
        while pending and pending[0].arrival <= time:
            waiting.append(pending.popleft())
        if mode != "static" or not running:
            while waiting and len(running) < capacity:
                running.append(waiting.popleft())
        executed = []
        for _ in range(min(token_budget, len(running))):
            job = running.popleft()
            first.setdefault(job.name, time + 1)
            remaining[job.name] -= 1
            executed.append(job.name)
            if not remaining[job.name]:
                finished[job.name] = time + 1
            else:
                running.append(job)
        trace.append({"round": time, "executed": executed,
                      "waiting": [j.name for j in waiting]})
        time += 1
    return {
        "kind": "discrete decode-round simulation; no prefill cost or GPU timing",
        "mode": mode, "rounds": time, "first_round": first, "finish_round": finished,
        "trace": trace,
    }
