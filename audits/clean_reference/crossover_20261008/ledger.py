"""Compare live RustEnv episode tuples with their JSON representation exactly."""
import math


def canonical_episode_ledger(ledger):
    """Normalize container types only; do not coerce or round numeric values."""
    if not isinstance(ledger, (list, tuple)):
        raise AssertionError('ledger must contain environment episode sequences')
    normalized = []
    for episodes in ledger:
        if not isinstance(episodes, (list, tuple)):
            raise AssertionError('invalid environment episode sequence')
        rows = []
        for row in episodes:
            if not isinstance(row, (list, tuple)) or len(row) != 5:
                raise AssertionError('invalid episode row')
            end_step, steps, reward, terminated, truncated = row
            if any(type(x) is not int for x in (end_step, steps, terminated, truncated)):
                raise AssertionError('episode counters and flags must be integers')
            if type(reward) is not float or not math.isfinite(reward):
                raise AssertionError('episode reward must be a finite float')
            if steps <= 0 or end_step < steps or terminated not in (0, 1) or truncated not in (0, 1):
                raise AssertionError('invalid episode counters or flags')
            if not (terminated or truncated):
                raise AssertionError('unfinished episode in completed ledger')
            rows.append(list(row))
        normalized.append(rows)
    return normalized


def assert_episode_ledger(live, expected):
    actual = canonical_episode_ledger(live)
    reference = canonical_episode_ledger(expected)
    if actual != reference:
        raise AssertionError('episode ledger numeric/order mismatch')
    return actual
