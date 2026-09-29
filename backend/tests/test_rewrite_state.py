from app.schemas.session import RewriteState


def _state(status: str, updated_at: float | None) -> RewriteState:
    return RewriteState(session_id="x" * 32, status=status, updated_at=updated_at)  # type: ignore[arg-type]


def test_running_fresh_heartbeat_is_not_stale():
    assert not _state("running", 1000.0).running_is_stale(now=1100.0, limit_sec=300)


def test_running_past_limit_is_stale():
    assert _state("running", 1000.0).running_is_stale(now=1400.0, limit_sec=300)


def test_running_without_heartbeat_is_stale():
    assert _state("running", None).running_is_stale(now=0.0, limit_sec=300)


def test_non_running_is_never_stale():
    assert not _state("ready", None).running_is_stale(now=1e12, limit_sec=1)
    assert not _state("failed", None).running_is_stale(now=1e12, limit_sec=1)
    assert not _state("idle", None).running_is_stale(now=1e12, limit_sec=1)
