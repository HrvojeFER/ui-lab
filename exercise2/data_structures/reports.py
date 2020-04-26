from abc import *
from typing import *

from data_structures.clause import Clause
from data_structures.resolvent import Resolvent
from data_structures.tasks import Task, Inquiry, PremiseRemoval, PremiseAddition


class Report(ABC):
    def short_str(self) -> str:
        return str(self)

    @abstractmethod
    def __str__(self) -> str:
        ...


class ResolutionReport(Report):
    _separator = '============='

    _premises: Tuple[Clause, ...]
    _goal: Clause
    _dependent_knowledge: Tuple[Resolvent, ...]
    _goal_is_true: Optional[bool]

    _negated_goal: Tuple[Clause, ...]
    _everything: Tuple[Clause, ...]

    @property
    def goal_is_true(self) -> bool:
        return self._goal_is_true if self._goal_is_true is not None else False

    @property
    def succeeded(self) -> bool:
        return self._goal_is_true is not None

    def __init__(self,
                 premises: Tuple[Clause, ...],
                 goal: Clause,
                 dependent_knowledge: Tuple[Resolvent, ...],
                 goal_is_true: Optional[bool]):
        self._premises = premises
        self._goal = goal
        self._dependent_knowledge = dependent_knowledge
        self._goal_is_true = goal_is_true

        self._negated_goal = tuple(-self._goal)
        self._everything = (*self._premises, *self._negated_goal, *self._dependent_knowledge)

    def short_str(self) -> str:
        return '%s is %s\n' % (str(self._goal), 'true' if self._goal_is_true else 'unknown')

    def __str__(self) -> str:
        if not self.goal_is_true:
            return self.short_str()

        clause_count = 1
        result = ''

        for premise in self._premises:
            result += '%d. %s\n' % (clause_count, str(premise))
            clause_count += 1

        result += ResolutionReport._separator + '\n'

        for clause in self._negated_goal:
            result += '%d. %s\n' % (clause_count, str(clause))
            clause_count += 1

        result += ResolutionReport._separator + '\n'

        for resolvent in self._dependent_knowledge:
            result += '%d. %s (%d, %d)\n' % (
                clause_count,
                str(resolvent),
                self._everything.index(resolvent.left_parent) + 1,
                self._everything.index(resolvent.right_parent) + 1)
            clause_count += 1

        result += ResolutionReport._separator + '\n'
        result += self.short_str()

        return result

    def __repr__(self) -> str:
        return str(self)


class TaskReport(Report, ABC):
    _task: Task
    _succeeded: bool

    def __init__(self, task: Task, succeeded: bool):
        self._task = task
        self._succeeded = succeeded


class PremiseRemovalReport(TaskReport):
    _task: PremiseRemoval

    def __str__(self) -> str:
        if self._succeeded:
            return 'removed %s' % (str(self._task.clause))
        else:
            return 'failed to remove %s' % (str(self._task.clause))


class PremiseAdditionReport(TaskReport):
    _task: PremiseAddition

    def __str__(self) -> str:
        if self._succeeded:
            return 'added %s' % (str(self._task.clause))
        else:
            return 'failed to add %s' % (str(self._task.clause))


class InquiryReport(TaskReport):
    _task: Inquiry
    _resolution_report: ResolutionReport

    def __init__(self, task: Task, resolution_report: ResolutionReport):
        super().__init__(task, resolution_report.succeeded)
        self._resolution_report = resolution_report

    def short_str(self) -> str:
        if self._succeeded:
            return str(self._resolution_report.short_str())
        else:
            return '%s is unknown' % str(self._task.clause)

    def __str__(self) -> str:
        if self._succeeded:
            return str(self._resolution_report)
        else:
            return '%s is unknown' % str(self._task.clause)
