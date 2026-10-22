"""Teams: routing, handoffs, failure containment and termination."""

from __future__ import annotations

from qwen_workshop.multiagent import CriticLoop, Message, Orchestrator, Router, Specialist, Team


def boom(_task: str) -> str:
    raise TimeoutError("upstream gone")


def team_of_two() -> Team:
    return Team().add(
        Specialist("researcher", "Reads the docs.", lambda t: f"docs:{t}"),
        Specialist("analyst", "Does arithmetic.", lambda t: "412.50"),
    )


def test_message_arrow_always_points_sender_to_recipient():
    rendered = str(Message("analyst", "orchestrator", "412.50", "result"))
    assert rendered.index("analyst") < rendered.index("orchestrator")
    assert "->" in rendered


def test_dispatch_records_both_directions():
    team = team_of_two()
    assert team.dispatch("user", "analyst", "total?") == "412.50"
    assert [m.kind for m in team.transcript] == ["task", "result"]
    assert team.n_handoffs == 2


def test_a_raising_specialist_does_not_kill_the_team():
    team = team_of_two().add(Specialist("flaky", "Breaks.", boom))
    result = team.dispatch("user", "flaky", "go")
    assert "TimeoutError" in result
    assert team.transcript[-1].kind == "error"
    # The team still works afterwards.
    assert team.dispatch("user", "analyst", "total?") == "412.50"


def test_dispatch_to_unknown_specialist_is_reported_not_raised():
    team = team_of_two()
    result = team.dispatch("user", "nobody", "go")
    assert "no such specialist" in result
    assert "researcher" in result  # tells you what IS available


def test_router_picks_from_the_roster():
    team = team_of_two()
    router = Router(team, lambda task, roster: "analyst")
    assert router.run("how much?") == "412.50"


def test_router_falls_back_visibly_on_an_invented_name():
    """Models invent plausible specialist names. That is routine, not fatal."""
    team = team_of_two()
    router = Router(team, lambda t, r: "DataScienceAgent", fallback="researcher")
    assert router.run("q").startswith("docs:")
    assert any(m.kind == "error" and "invalid choice" in m.content
               for m in team.transcript)


def test_router_without_fallback_says_so():
    team = team_of_two()
    assert "No specialist" in Router(team, lambda t, r: "nope").run("q")


def test_roster_is_what_the_router_reads():
    roster = team_of_two().roster()
    assert "researcher: Reads the docs." in roster
    assert "analyst: Does arithmetic." in roster


def test_orchestrator_gathers_then_synthesises():
    team = team_of_two()
    orchestrator = Orchestrator(
        team, lambda goal, results: f"{goal}|{sorted(results)}")
    answer = orchestrator.run("combined", [("analyst", "sum"), ("researcher", "read")])
    assert "analyst" in answer and "researcher" in answer
    assert team.n_handoffs == 5  # 2 tasks + 2 results + 1 final


def test_orchestrator_survives_one_broken_worker():
    team = team_of_two().add(Specialist("flaky", "Breaks.", boom))
    got = {}
    orchestrator = Orchestrator(team, lambda g, r: got.update(r) or "done")
    orchestrator.run("goal", [("analyst", "sum"), ("flaky", "x")])
    assert got["analyst"] == "412.50"
    assert "TimeoutError" in got["flaky"]  # the failure reaches the synthesiser


def test_critic_loop_stops_on_the_accept_token():
    drafts = iter(["v1", "v2", "v3"])

    team = Team().add(
        Specialist("writer", "writes", lambda t: next(drafts)),
        Specialist("critic", "reviews", lambda d: "APPROVED" if d == "v3" else "more"),
    )
    outcome = CriticLoop(team, "writer", "critic", max_rounds=5).run("write")
    assert outcome["accepted"] is True
    assert outcome["rounds"] == 3


def test_critic_loop_terminates_when_never_satisfied():
    """The characteristic multi-agent failure: polite forever."""
    team = Team().add(
        Specialist("writer", "writes", lambda t: "draft"),
        Specialist("critic", "reviews", lambda d: "Needs work."),
    )
    outcome = CriticLoop(team, "writer", "critic", max_rounds=2).run("write")
    assert outcome["accepted"] is False
    assert outcome["rounds"] == 2


def test_hedged_approval_does_not_count():
    """'Looks good, although...' is not an acceptance."""
    team = Team().add(
        Specialist("writer", "writes", lambda t: "draft"),
        Specialist("critic", "reviews", lambda d: "Looks good to me, although the tone is off."),
    )
    assert CriticLoop(team, "writer", "critic", max_rounds=1).run("w")["accepted"] is False


def test_critique_is_tagged_in_the_transcript():
    team = Team().add(
        Specialist("writer", "writes", lambda t: "draft"),
        Specialist("critic", "reviews", lambda d: "APPROVED"),
    )
    CriticLoop(team, "writer", "critic").run("w")
    assert any(m.kind == "critique" for m in team.transcript)
