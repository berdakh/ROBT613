"""Plans: validation, wave ordering, failure propagation and replanning."""

from __future__ import annotations

import pytest

from qwen_workshop.planning import Plan, PlanError, Task, topological_order


def simple_rows():
    return [
        {"id": "a", "description": "search"},
        {"id": "b", "description": "search again"},
        {"id": "c", "description": "read a", "depends_on": ["a"]},
        {"id": "d", "description": "combine", "depends_on": ["b", "c"]},
    ]


def test_waves_group_independent_tasks():
    plan = Plan.from_dicts("goal", simple_rows())
    assert plan.waves() == [["a", "b"], ["c"], ["d"]]


def test_cycle_is_rejected():
    with pytest.raises(PlanError, match="cycle"):
        Plan.from_dicts("g", [
            {"id": "a", "description": "x", "depends_on": ["b"]},
            {"id": "b", "description": "y", "depends_on": ["a"]},
        ])


def test_unknown_dependency_is_rejected():
    with pytest.raises(PlanError, match="unknown task"):
        Plan.from_dicts("g", [{"id": "a", "description": "x", "depends_on": ["ghost"]}])


def test_duplicate_ids_are_rejected():
    with pytest.raises(PlanError, match="duplicate"):
        topological_order([Task("a", "x"), Task("a", "y")])


def test_empty_plan_is_rejected():
    with pytest.raises(PlanError):
        Plan.from_dicts("g", [])


def test_task_without_description_is_rejected():
    with pytest.raises(PlanError, match="description"):
        Plan.from_dicts("g", [{"id": "a"}])


def test_execute_runs_everything_and_passes_dependency_results():
    plan = Plan.from_dicts("goal", simple_rows())
    seen = {}

    def runner(task, inputs):
        seen[task.id] = dict(inputs)
        return f"result-{task.id}"

    plan.execute(runner)
    assert plan.complete
    assert seen["d"] == {"b": "result-b", "c": "result-c"}
    assert plan.results()["d"] == "result-d"


def test_failure_blocks_downstream_but_not_siblings():
    plan = Plan.from_dicts("goal", simple_rows())

    def runner(task, inputs):
        if task.id == "a":
            raise FileNotFoundError("missing")
        return "ok"

    plan.execute(runner)
    assert [t.id for t in plan.failed] == ["a"]
    assert sorted(t.id for t in plan.blocked) == ["c", "d"]
    # b was independent of the failure and must still have run.
    assert plan.by_id["b"].status == "done"
    assert plan.complete


def test_blocked_tasks_say_what_blocked_them():
    plan = Plan.from_dicts("g", [
        {"id": "a", "description": "x"},
        {"id": "b", "description": "y", "depends_on": ["a"]},
    ])
    plan.execute(lambda t, i: (_ for _ in ()).throw(ValueError("no")))
    assert "blocked by a" in plan.by_id["b"].error


def test_stop_on_failure_abandons_remaining_work():
    plan = Plan.from_dicts("g", [
        {"id": "a", "description": "x"},
        {"id": "b", "description": "y"},
        {"id": "c", "description": "z"},
    ])
    plan.execute(lambda t, i: (_ for _ in ()).throw(ValueError("no")),
                 stop_on_failure=True)
    assert len(plan.failed) == 1
    assert len([t for t in plan.tasks if t.status == "pending"]) == 2


def test_replan_keeps_completed_work():
    plan = Plan.from_dicts("goal", simple_rows())
    plan.execute(lambda t, i: "ok" if t.id != "c" else (_ for _ in ()).throw(OSError("x")))
    assert plan.by_id["a"].status == "done"

    plan.replan_after_failure(lambda p, stuck: [
        {"id": "c2", "description": "read a differently", "depends_on": ["a"]},
        {"id": "d2", "description": "combine", "depends_on": ["b", "c2"]},
    ])
    assert {t.id for t in plan.tasks} == {"a", "b", "c2", "d2"}
    assert plan.by_id["a"].result == "ok"  # not re-run
    assert plan.replans == 1
    plan.execute(lambda t, i: "ok")
    assert plan.complete and not plan.failed


def test_replan_cannot_depend_on_the_wreckage():
    plan = Plan.from_dicts("g", [{"id": "a", "description": "x"}])
    plan.execute(lambda t, i: (_ for _ in ()).throw(ValueError("no")))
    with pytest.raises(PlanError, match="neither completed work"):
        plan.replan_after_failure(lambda p, stuck: [
            {"id": "b", "description": "y", "depends_on": ["a"]}])


def test_replan_with_no_failures_is_a_noop():
    plan = Plan.from_dicts("g", [{"id": "a", "description": "x"}])
    plan.execute(lambda t, i: "ok")
    plan.replan_after_failure(lambda p, stuck: [{"id": "z", "description": "never"}])
    assert [t.id for t in plan.tasks] == ["a"]


def test_giving_up_is_a_legitimate_replan():
    plan = Plan.from_dicts("g", [{"id": "a", "description": "x"}])
    plan.execute(lambda t, i: (_ for _ in ()).throw(ValueError("no")))
    plan.replan_after_failure(lambda p, stuck: [])
    assert plan.tasks == []


def test_parse_plan_out_of_chatty_model_output():
    text = (
        "Sure, here's the plan!\n```json\n"
        '{"tasks": [{"id": "x", "description": "do x"},'
        ' {"id": "y", "description": "do y", "depends_on": ["x"]}]}\n'
        "```\nLet me know if that works."
    )
    plan = Plan.from_model_output("g", text)
    assert plan.waves() == [["x"], ["y"]]


def test_parse_accepts_a_bare_list():
    plan = Plan.from_model_output("g", '[{"id": "x", "description": "do x"}]')
    assert len(plan.tasks) == 1


def test_unparseable_plan_raises_rather_than_doing_nothing():
    with pytest.raises(PlanError, match="no task list"):
        Plan.from_model_output("g", "I'm not sure how to break this down.")
