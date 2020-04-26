from typing import *

from data_structures.clause import Clause
from data_structures.resolvent import Resolvent
from data_structures.tasks import Task, Inquiry, PremiseRemoval, PremiseAddition
from data_structures.reports import \
    ResolutionReport, TaskReport, InquiryReport, PremiseRemovalReport, PremiseAdditionReport

# noinspection PyUnresolvedReferences
from algorithms.control_strategies import ControlStrategy, SaturationByLevels, SupportSetStrategy
from algorithms.simplification_strategies import RedundantClauseRemoval, InsignificantClauseRemoval


class RefutationResolution:
    _control_strategy: ControlStrategy

    _resolve_clauses: Callable[[Clause, Clause], Resolvent]

    @property
    def premises(self) -> FrozenSet[Clause]:
        return self._control_strategy.premises

    @property
    def acquired_knowledge(self) -> Tuple[Resolvent, ...]:
        return self._control_strategy.acquired_knowledge

    @property
    def goal(self) -> Clause:
        return self._control_strategy.goal

    def __init__(self, control_strategy: ControlStrategy):
        self._control_strategy: ControlStrategy = control_strategy

        if self._control_strategy.removes_insignificant_clauses():
            self._resolve_clauses: Callable[[Clause, Clause], Resolvent] = RefutationResolution._resolve_non_tautologies
        else:
            self._resolve_clauses: Callable[[Clause, Clause], Resolvent] = \
                RefutationResolution._resolve_possible_tautologies

    def resolve(self, goal: Optional[Clause] = None, keep_acquired_knowledge: bool = False) -> ResolutionReport:
        # noinspection PyProtectedMember
        """
        >>> premises = {Clause.parse('a'), Clause.parse('b v ~a'), Clause.parse('~b v c')}
        >>> goal = Clause.parse('c')
        >>> simplification_strategies = {RedundantClauseRemoval(), InsignificantClauseRemoval()}
        >>> control_strategy = SupportSetStrategy(premises, goal, simplification_strategies)
        >>> resolution = RefutationResolution(control_strategy)
        >>> resolution.resolve().goal_is_true
        True
        """
        self._control_strategy.prepare_for_resolution(goal, keep_acquired_knowledge)

        while True:
            for clause_pair in self._control_strategy.get_new_clause_pairs():
                resolvent = self._resolve_clauses(*clause_pair)

                if resolvent.is_nil():
                    return self._write_report(final_resolvent=resolvent)

                if resolvent.is_new_knowledge():
                    self._control_strategy.acquire_temporary_knowledge(resolvent)

            if not self._control_strategy.found_any_new_knowledge():
                return self._write_report()

            self._control_strategy.consolidate_knowledge()

    def do(self, task: Task) -> TaskReport:
        if isinstance(task, Inquiry):
            resolution_report = self.resolve(task.clause)
            return InquiryReport(task, resolution_report)
        elif isinstance(task, PremiseRemoval):
            return PremiseRemovalReport(task, succeeded=self._control_strategy.remove_premise(task.clause).succeeded)
        elif isinstance(task, PremiseAddition):
            return PremiseAdditionReport(task, succeeded=self._control_strategy.add_premise(task.clause).succeeded)

    def _write_report(self, final_resolvent: Optional[Resolvent] = None) -> ResolutionReport:
        premises = self._control_strategy.premises
        goal = self._control_strategy.goal
        if final_resolvent is not None:
            dependent_knowledge = (*self._get_dependent_knowledge(final_resolvent), )
            goal_is_true = True
        else:
            dependent_knowledge = self._control_strategy.acquired_knowledge
            goal_is_true = False

        return ResolutionReport(tuple(premises), goal, dependent_knowledge, goal_is_true)

    def _get_dependent_knowledge(self, clause: Clause) -> Tuple[Resolvent, ...]:
        if isinstance(clause, Resolvent):
            return (*self._get_dependent_knowledge(clause.left_parent),
                    *self._get_dependent_knowledge(clause.right_parent),
                    clause)

        return ()

    @staticmethod
    def _resolve_possible_tautologies(left_clause: Clause, right_clause: Clause) -> Resolvent:
        # noinspection PyProtectedMember
        """
        >>> left_clause = Clause.parse('a v ~e v c v d v ~d')
        >>> right_clause = Clause.parse('~a v e v b v d v ~d')
        >>> str(RefutationResolution._resolve_possible_tautologies(left_clause, right_clause))
        'b v c'

        >>> left_clause = Clause.parse('c v d')
        >>> right_clause = Clause.parse('b v d')
        >>> str(RefutationResolution._resolve_non_tautologies(left_clause, right_clause))
        'b v c v d'
        """
        return Resolvent(Clause.remove_complementary_literals(left_clause | right_clause)) \
            .set_parents(left_clause, right_clause)

    @staticmethod
    def _resolve_non_tautologies(left_clause: Clause, right_clause: Clause) -> Resolvent:
        # noinspection PyProtectedMember
        """
        >>> left_clause = Clause.parse('a v ~e v c v d')
        >>> right_clause = Clause.parse('~a v e v b v d')
        >>> str(RefutationResolution._resolve_non_tautologies(left_clause, right_clause))
        'b v c v d'

        >>> left_clause = Clause.parse('c v d')
        >>> right_clause = Clause.parse('b v d')
        >>> str(RefutationResolution._resolve_non_tautologies(left_clause, right_clause))
        'b v c v d'
        """
        return Resolvent(
            left_clause & right_clause | Clause.remove_complementary_literals(left_clause ^ right_clause)) \
            .set_parents(left_clause, right_clause)
