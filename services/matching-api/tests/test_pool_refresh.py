import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pool_refresh import refresh_pool  # noqa: E402


class FakeClock:
    """Horloge et sommeil factices : `sleep` fait avancer `now` sans attendre."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = 0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.sleeps += 1


def test_runs_a_pass_immediately_when_nothing_is_running():
    fake = FakeClock()
    passes = []

    ok = refresh_pool(
        lambda: False,
        lambda: passes.append(1),
        max_wait_seconds=60,
        sleep=fake.sleep,
        clock=fake.clock,
    )

    assert ok is True
    assert passes == [1]
    assert fake.sleeps == 0


def test_waits_for_a_running_pass_to_finish_before_starting_its_own():
    # Un passage déjà en cours a pu démarrer avant l'ajout des nouveaux CV :
    # il doit finir, puis un nouveau passage doit être lancé.
    fake = FakeClock()
    events = []
    running_polls = {"left": 3}

    def is_running() -> bool:
        if running_polls["left"] > 0:
            running_polls["left"] -= 1
            return True
        return False

    ok = refresh_pool(
        is_running,
        lambda: events.append("pass"),
        max_wait_seconds=60,
        sleep=fake.sleep,
        clock=fake.clock,
    )

    assert ok is True
    assert events == ["pass"]
    assert fake.sleeps == 3


def test_gives_up_without_running_a_pass_if_the_running_one_never_finishes():
    fake = FakeClock()
    passes = []

    ok = refresh_pool(
        lambda: True,
        lambda: passes.append(1),
        max_wait_seconds=10,
        poll_seconds=1,
        sleep=fake.sleep,
        clock=fake.clock,
    )

    assert ok is False
    assert passes == []
    assert fake.now >= 10


def test_waits_again_if_another_pass_slipped_in_after_ours():
    # Notre appel a été ignoré (run_full_reindex rend la main si un passage
    # tourne déjà) : on ne doit pas scorer pendant que l'autre écrit encore.
    fake = FakeClock()
    state = {"running": False, "polls_after_pass": 2}

    def is_running() -> bool:
        if state["running"] and state["polls_after_pass"] > 0:
            state["polls_after_pass"] -= 1
            return True
        return False

    def run_pass() -> None:
        state["running"] = True  # un autre passage prend la place

    ok = refresh_pool(
        is_running,
        run_pass,
        max_wait_seconds=60,
        sleep=fake.sleep,
        clock=fake.clock,
    )

    assert ok is True
    assert fake.sleeps == 2


def test_reports_false_if_a_competing_pass_outlasts_the_deadline():
    fake = FakeClock()
    state = {"started": False}

    def is_running() -> bool:
        return state["started"]

    def run_pass() -> None:
        state["started"] = True  # et ne finit jamais

    ok = refresh_pool(
        is_running,
        run_pass,
        max_wait_seconds=5,
        poll_seconds=1,
        sleep=fake.sleep,
        clock=fake.clock,
    )

    assert ok is False
