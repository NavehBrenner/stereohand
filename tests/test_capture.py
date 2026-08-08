"""Skew-decision predicate — the cv2-free core of capture (runs in CI)."""

from __future__ import annotations

from stereohand.capture import pair_status, within_skew


def test_within_threshold_accepted():
    assert within_skew(100.000, 100.010, max_skew_s=0.02) is True


def test_over_threshold_rejected():
    assert within_skew(100.000, 100.050, max_skew_s=0.02) is False


def test_boundary_is_inclusive():
    assert within_skew(100.000, 100.020, max_skew_s=0.02) is True


def test_order_does_not_matter():
    assert within_skew(100.05, 100.0, max_skew_s=0.02) == within_skew(
        100.0, 100.05, max_skew_s=0.02
    )


# -- pair_status: the one decision read() and latest_pair_timestamp() share ---------------

_LIMITS = {"max_skew_s": 0.02, "max_age_s": 0.5}


def test_fresh_and_synced_is_ok():
    assert pair_status(100.0, 100.01, now=100.02, **_LIMITS) == "ok"


def test_skew_boundary_is_inclusive_like_within_skew():
    assert pair_status(100.0, 100.02, now=100.03, **_LIMITS) == "ok"


def test_out_of_phase_is_over_skew_not_absent():
    assert pair_status(100.0, 100.03, now=100.04, **_LIMITS) == "over_skew"


def test_stalled_camera_is_stale():
    assert pair_status(100.0, 100.0, now=100.6, **_LIMITS) == "stale"


def test_stale_outranks_over_skew():
    """A camera that stopped delivering also drifts out of skew.

    If that were reported as ``over_skew`` the consumer would hold its last reading forever,
    because holding is exactly what ``over_skew`` asks for. ``stale`` is the backstop that
    makes holding safe, so it has to win when both are true.
    """
    assert pair_status(100.0, 101.0, now=101.01, **_LIMITS) == "stale"


def _newest_pair(now: float, period: float, phase: float) -> tuple[float, float]:
    """Newest frame timestamp from each of two free-running cameras at wall-clock `now`."""
    newest_left = (now // period) * period
    newest_right = ((now - phase) // period) * period + phase
    return newest_left, newest_right


def test_free_running_camera_skew_is_bimodal_never_the_average():
    """The mechanism behind the bug: sampled skew takes exactly two values, not a spread.

    Two uncoordinated cameras at period T with phase offset phi never sit at the *average*
    skew. Taking the newest frame from each, the difference is `phi` for part of every cycle
    and `T - phi` for the rest, because for that stretch one camera has ticked and the other
    has not caught up yet. So a tolerance comfortably above the observed *mean* can still
    reject a large and fixed fraction of every cycle — which is why "mean skew 13.5 ms
    against a 20 ms tolerance" read as healthy while a third of pairs were being dropped.
    """
    period, phase = 0.0334, 0.0135
    observed = {
        round(abs(left - right), 6)
        for left, right in (
            _newest_pair(step * period / 40.0, period, phase) for step in range(400)
        )
    }
    assert observed == {round(phase, 6), round(period - phase, 6)}


def test_a_tolerance_between_the_two_modes_yields_both_ok_and_over_skew():
    """The over-skew stretch must classify as `over_skew` (hold), never `stale` (absent).

    Deliberately *not* the dev rig's exact numbers. There T ~ 33.4 ms and phi ~ 13.5 ms put
    `T - phi` at 19.9 ms, a hair under the 20 ms default — so the rejection rate sat right on
    the boundary and swung between 24% and 41% across runs of the same probe. Pinning a
    knife-edge would make this test a coin flip; the mechanism is what matters, so the
    tolerance here sits clearly between the two modes.

    The consumer contract is what turns this into correct behaviour — see
    test_over_skew_holds_previous_reading in test_tracker.py.
    """
    period, phase = 0.0334, 0.0135
    limits = {"max_skew_s": 0.016, "max_age_s": 0.5}  # between phi (13.5) and T-phi (19.9)
    statuses = set()
    for step in range(400):
        now = step * period / 40.0
        left, right = _newest_pair(now, period, phase)
        statuses.add(pair_status(left, right, now=now, **limits))

    assert statuses == {"ok", "over_skew"}, (
        f"expected the cycle to alternate ok/over_skew, got {sorted(statuses)}"
    )
