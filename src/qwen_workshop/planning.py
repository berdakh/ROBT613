"""Planning: deciding the steps up front instead of one at a time.

Two ways to run a multi-step task
---------------------------------
**ReAct** (notebook 10) decides one step at a time. Flexible, and it adapts to
whatever it just found - but nothing ever checks that the steps add up to the
task, so a ReAct agent can wander for six steps and answer a question nobody
asked.

**Plan-then-execute** writes the whole plan first, then runs it. You get three
things ReAct cannot give you:

1. **The plan is inspectable before anything runs.** You can print it, cost
   it, and refuse it. This is the only point at which a human can intervene
   cheaply.
2. **Independent steps can run in parallel**, because the dependencies are
   written down rather than implied by order.
3. **Failure is localised.** When step 3 fails you know steps 4 and 5 depended
   on it, so you block those instead of running them on garbage.

The cost is rigidity: a plan written before step 1 ran is a plan written in
ignorance. That is what :meth:`Plan.replan_after_failure` is for, and why real
systems are a loop of plan → execute → replan rather than either extreme.

Nothing here calls a model. A plan is data; this module is the data structure
and the executor, which means it can be tested exactly like any other code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = ["Plan", "PlanError", "Task", "topological_order"]


class PlanError(Exception):
    """A plan that cannot be executed - a cycle, a missing dependency, a dud."""


@dataclass
class Task:
    """One step. `depends_on` is what makes this a plan and not a list.

    Status moves ``pending -> running -> done | failed | blocked``. ``blocked``
    is the one people forget to model, and then they run step 4 on the output
    of a step 3 that never produced any.
    """

    id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"
    result: str | None = None
    error: str | None = None

    @property
    def finished(self) -> bool:
        return self.status in ("done", "failed", "blocked")

    def __str__(self) -> str:
        mark = {"pending": "·", "running": ">", "done": "✓",
                "failed": "✗", "blocked": "∅"}[self.status]
        deps = f"  (needs {', '.join(self.depends_on)})" if self.depends_on else ""
        return f"{mark} {self.id}: {self.description}{deps}"


def topological_order(tasks: list[Task]) -> list[list[str]]:
    """Group task ids into waves that may run together.

    Wave 0 is everything with no dependencies; wave 1 is everything whose
    dependencies are all in wave 0, and so on. The *number of waves* is the
    shortest the plan can possibly take, which is a useful thing to know
    before you start paying for it.

    Raises:
        PlanError: on a cycle or an unknown dependency.
    """
    by_id = {t.id: t for t in tasks}
    if len(by_id) != len(tasks):
        raise PlanError("duplicate task ids")
    for task in tasks:
        for dep in task.depends_on:
            if dep not in by_id:
                raise PlanError(f"task {task.id!r} depends on unknown task {dep!r}")

    waves: list[list[str]] = []
    placed: set[str] = set()
    remaining = list(by_id)
    while remaining:
        wave = [tid for tid in remaining if set(by_id[tid].depends_on) <= placed]
        if not wave:
            raise PlanError(
                f"dependency cycle among {sorted(remaining)} - no task can start"
            )
        waves.append(wave)
        placed.update(wave)
        remaining = [tid for tid in remaining if tid not in placed]
    return waves


@dataclass
class Plan:
    """A set of tasks with dependencies, plus the executor that runs them."""

    goal: str
    tasks: list[Task] = field(default_factory=list)
    replans: int = 0

    # -- construction -------------------------------------------------------

    @classmethod
    def from_dicts(cls, goal: str, rows: list[dict]) -> Plan:
        """Build from plain dicts - typically straight out of a model's JSON.

        Validates on the way in. A malformed plan should fail here, loudly,
        not three steps into execution.
        """
        if not rows:
            raise PlanError("a plan with no tasks is not a plan")
        tasks = []
        for i, row in enumerate(rows):
            if "description" not in row:
                raise PlanError(f"task {i} has no 'description'")
            tasks.append(
                Task(
                    id=str(row.get("id", f"t{i + 1}")),
                    description=str(row["description"]),
                    depends_on=[str(d) for d in row.get("depends_on", [])],
                )
            )
        plan = cls(goal=goal, tasks=tasks)
        plan.validate()
        return plan

    @classmethod
    def from_model_output(cls, goal: str, text: str) -> Plan:
        """Parse a plan out of whatever the model actually said.

        Models wrap JSON in prose and code fences; `extract_json` copes.
        A model that produces no parseable plan is a real failure mode, so it
        raises rather than returning an empty plan that silently does nothing.
        """
        from .parsing import extract_json

        data = extract_json(text)
        if isinstance(data, dict):
            data = data.get("tasks") or data.get("plan") or data.get("steps")
        if not isinstance(data, list):
            raise PlanError(
                "no task list found in the model output. Got:\n" + text[:300]
            )
        return cls.from_dicts(goal, data)

    def validate(self) -> Plan:
        topological_order(self.tasks)  # raises on cycles / unknown deps
        return self

    # -- inspection ---------------------------------------------------------

    @property
    def by_id(self) -> dict[str, Task]:
        return {t.id: t for t in self.tasks}

    def waves(self) -> list[list[str]]:
        return topological_order(self.tasks)

    def ready(self) -> list[Task]:
        """Pending tasks whose dependencies have all completed successfully."""
        done = {t.id for t in self.tasks if t.status == "done"}
        return [
            t for t in self.tasks
            if t.status == "pending" and set(t.depends_on) <= done
        ]

    def render(self) -> str:
        waves = self.waves()
        lines = [f"GOAL: {self.goal}",
                 f"({len(self.tasks)} tasks in {len(waves)} waves"
                 + (f", {self.replans} replans" if self.replans else "") + ")"]
        for i, wave in enumerate(waves):
            lines.append(f"  wave {i}:")
            for tid in wave:
                lines.append(f"    {self.by_id[tid]}")
        return "\n".join(lines)

    def results(self) -> dict[str, str]:
        return {t.id: t.result for t in self.tasks if t.status == "done" and t.result}

    @property
    def failed(self) -> list[Task]:
        return [t for t in self.tasks if t.status == "failed"]

    @property
    def blocked(self) -> list[Task]:
        return [t for t in self.tasks if t.status == "blocked"]

    @property
    def complete(self) -> bool:
        return all(t.finished for t in self.tasks)

    # -- execution ----------------------------------------------------------

    def execute(
        self,
        runner: Callable[[Task, dict[str, str]], str],
        *,
        on_task: Callable[[Task], None] | None = None,
        stop_on_failure: bool = False,
    ) -> Plan:
        """Run every task whose dependencies succeeded.

        Args:
            runner: does the work. Takes the task and the results of its
                dependencies, returns a result string. Raise to fail the task -
                this is the one place an exception is the right interface,
                because a tool that half-worked should not look like success.
            on_task: called after each task finishes. For progress output.
            stop_on_failure: abandon the run at the first failure instead of
                finishing the independent branches. Default False, because
                partial results from the branches that did work are usually
                worth having.

        Downstream tasks of a failure become ``blocked``, never ``failed``:
        they did not fail, they never ran, and conflating the two makes a
        trace unreadable.
        """
        while True:
            ready = self.ready()
            if not ready:
                break
            for task in ready:
                task.status = "running"
                inputs = {dep: self.by_id[dep].result or "" for dep in task.depends_on}
                try:
                    task.result = runner(task, inputs)
                    task.status = "done"
                except Exception as exc:  # noqa: BLE001 - the runner may raise anything
                    task.status = "failed"
                    task.error = f"{type(exc).__name__}: {exc}"
                if on_task:
                    on_task(task)
                if task.status == "failed" and stop_on_failure:
                    self._block_unreachable()
                    return self
        self._block_unreachable()
        return self

    def _block_unreachable(self) -> None:
        """Anything still pending can never run. Say so explicitly."""
        changed = True
        while changed:
            changed = False
            bad = {t.id for t in self.tasks if t.status in ("failed", "blocked")}
            for task in self.tasks:
                if task.status == "pending" and set(task.depends_on) & bad:
                    task.status = "blocked"
                    blocker = sorted(set(task.depends_on) & bad)
                    task.error = f"blocked by {', '.join(blocker)}"
                    changed = True

    # -- recovery -----------------------------------------------------------

    def replan_after_failure(
        self, planner: Callable[[Plan, list[Task]], list[dict]]
    ) -> Plan:
        """Ask `planner` for replacement tasks for the failed and blocked ones.

        The completed tasks and their results are kept - replanning from
        scratch throws away work that was fine, which is both slower and a
        good way to loop forever.

        `planner` receives this plan and the tasks that did not run, and
        returns new task dicts. Returning ``[]`` means "give up", which is a
        legitimate answer and better than inventing a step.
        """
        stuck = self.failed + self.blocked
        if not stuck:
            return self
        rows = planner(self, stuck)
        survivors = [t for t in self.tasks if t.status == "done"]
        done_ids = {t.id for t in survivors}

        new_tasks = []
        for i, row in enumerate(rows):
            tid = str(row.get("id", f"r{self.replans + 1}_{i + 1}"))
            deps = [str(d) for d in row.get("depends_on", [])]
            # A replacement task may legitimately depend on work that already
            # succeeded; it may not depend on the wreckage it is replacing.
            bad = [d for d in deps if d not in done_ids and
                   d not in {str(r.get("id", f"r{self.replans + 1}_{j + 1}"))
                             for j, r in enumerate(rows)}]
            if bad:
                raise PlanError(
                    f"replacement task {tid!r} depends on {bad}, which is neither "
                    "completed work nor part of the new plan"
                )
            new_tasks.append(Task(id=tid, description=str(row["description"]),
                                  depends_on=deps))

        self.tasks = survivors + new_tasks
        self.replans += 1
        self.validate()
        return self
