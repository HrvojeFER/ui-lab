from abc import *
from typing import *
from dataclasses import *
import itertools

from data_structures.clause import Clause
from data_structures.resolvent import Resolvent

from algorithms.simplification_strategies import SimplificationStrategy, InsignificantClauseRemoval


@dataclass(frozen=True)
class PremiseMutationReport:
    succeeded: bool


class ControlStrategy(ABC):
    _premises: Set[Clause]

    _goal: Clause
    _negated_goal: Set[Clause]

    _resolution_clauses: Set[Clause]
    _temporary_acquired_knowledge: Set[Resolvent]
    _acquired_knowledge: Set[Resolvent]

    _simplification_strategies: Tuple[SimplificationStrategy, ...]

    @property
    def premises(self) -> FrozenSet[Clause]:
        return frozenset(self._premises)

    @property
    def acquired_knowledge(self) -> Tuple[Resolvent, ...]:
        return tuple(self._acquired_knowledge)

    @property
    def goal(self) -> Clause:
        return self._goal

    def __init__(self,
                 premises: Iterable[Clause],
                 goal: Optional[Clause] = None,
                 simplification_strategies: Optional[Iterable[SimplificationStrategy]] = None):
        if isinstance(premises, set):
            self._premises = premises
        else:
            self._premises = set(premises)

        if goal is not None:
            self._goal = goal
            self._negated_goal = set(-self._goal)

        self._resolution_clauses = set()
        self._temporary_acquired_knowledge = set()
        self._acquired_knowledge = set()

        if simplification_strategies is not None:
            self._simplification_strategies = tuple(simplification_strategies)
        else:
            self._simplification_strategies = tuple()

    # noinspection PyArgumentList
    def add_premise(self, premise: Clause) -> PremiseMutationReport:
        self._premises.add(premise)
        return PremiseMutationReport(True)

    # noinspection PyArgumentList
    def remove_premise(self, premise: Clause) -> PremiseMutationReport:
        if premise in self._premises:
            self._remove_dependent_knowledge(premise)
            return PremiseMutationReport(True)

        return PremiseMutationReport(False)

    def _remove_dependent_knowledge(self, *knowledge: Union[Tuple[Clause, ...], Clause]) -> NoReturn:
        # noinspection PyProtectedMember
        """
        >>> strategy = SaturationByLevels({Clause.parse('a'), Clause.parse('b')}, Clause.parse('g'))
        >>> strategy._acquired_knowledge = \
                { \
                    Resolvent(Clause.parse('c')).set_parents(Clause.parse('a'), Clause.parse('b')), \
                    Resolvent(Clause.parse('d')).set_parents(Clause.parse('c'), Clause.parse('b')) \
                }
        >>> strategy.remove_premise(Clause.parse('znj')).succeeded
        False

        >>> report = strategy.remove_premise(Clause.parse('a'))
        >>> len(strategy._acquired_knowledge)
        0
        """

        knowledge = frozenset(knowledge)
        for clause in knowledge:
            if clause in self._premises:
                self._premises.remove(clause)
            elif clause in self._acquired_knowledge:
                assert isinstance(clause, Resolvent)
                self._acquired_knowledge.remove(clause)

        dependent_knowledge = frozenset(filter(
            lambda resolvent: resolvent.right_parent in knowledge or resolvent.left_parent in knowledge,
            self._acquired_knowledge))
        if len(dependent_knowledge) > 0:
            self._remove_dependent_knowledge(*dependent_knowledge)

    def removes_insignificant_clauses(self) -> bool:
        for simplification_strategy in self._simplification_strategies:
            if isinstance(simplification_strategy, InsignificantClauseRemoval):
                return True

        return False

    def prepare_for_resolution(self, goal: Optional[Clause] = None, keep_acquired_knowledge: bool = False) -> None:
        if goal is not None:
            self._goal = goal
            self._negated_goal = set(-self._goal)

        if not keep_acquired_knowledge:
            self._acquired_knowledge = set()
        self._temporary_acquired_knowledge = set()

        self._resolution_clauses = set(self._premises | set(self._acquired_knowledge) | set(-self._goal))
        self.simplify_resolution_clauses()

    def found_any_new_knowledge(self) -> bool:
        """
        >>> strategy = SaturationByLevels({Clause.parse('a'), Clause.parse('b')}, Clause.parse('g'))
        >>> strategy.prepare_for_resolution()
        >>> strategy._temporary_acquired_knowledge = {Clause.parse('n1'), Clause.parse('n2')}
        >>> strategy.found_any_new_knowledge()
        True
        """

        return not self._temporary_acquired_knowledge <= self._resolution_clauses

    def acquire_temporary_knowledge(self, *resolvents: Union[Tuple[Resolvent, ...], Resolvent]) -> NoReturn:
        self._temporary_acquired_knowledge |= set(resolvents)

    def consolidate_knowledge(self) -> NoReturn:
        self._resolution_clauses |= self._temporary_acquired_knowledge
        self.simplify_resolution_clauses()

        for resolvent in self._temporary_acquired_knowledge:
            if resolvent in self._resolution_clauses:
                self._acquired_knowledge.add(resolvent)

        self._temporary_acquired_knowledge = set()

    def simplify_resolution_clauses(self) -> NoReturn:
        for simplification_strategy in self._simplification_strategies:
            simplified = simplification_strategy.simplify(self._resolution_clauses)
            if isinstance(simplified, set):
                self._resolution_clauses = simplified
            else:
                self._resolution_clauses = set(simplified)

    @abstractmethod
    def get_new_clause_pairs(self) -> Iterable[Tuple[Clause, Clause]]:
        ...

    def __str__(self) -> str:
        return 'Clauses: %s \n Goal: %s \n New premises: %s \n Resolution premises: %s \n' % \
               (str(self._premises), str(self._goal), str(self._acquired_knowledge), str(self._resolution_clauses))

    def __repr__(self) -> str:
        return str(self)


class SaturationByLevels(ControlStrategy):
    def get_new_clause_pairs(self) -> Iterable[Tuple[Clause, Clause]]:
        """
        >>> strategy = SaturationByLevels({Clause.parse('a'), Clause.parse('b')}, Clause.parse('g'))
        >>> strategy.prepare_for_resolution()
        >>> strategy._temporary_acquired_knowledge = {Clause.parse('n1'), Clause.parse('n2')}
        >>> strategy.consolidate_knowledge()
        >>> len(set(strategy.get_new_clause_pairs()))
        10
        """
        # noinspection PyTypeChecker

        return itertools.combinations(self._resolution_clauses, 2)


class SupportSetStrategy(ControlStrategy):
    @property
    def support_set(self) -> Set[Clause]:
        return self._resolution_clauses & set(self._acquired_knowledge) | self._resolution_clauses & set(-self._goal)

    def get_new_clause_pairs(self) -> Iterable[Tuple[Clause, Clause]]:
        """
        >>> strategy = SupportSetStrategy({Clause.parse('a'), Clause.parse('b')}, Clause.parse('g'))
        >>> strategy.prepare_for_resolution()
        >>> strategy._temporary_acquired_knowledge = {Clause.parse('n1'), Clause.parse('n2')}
        >>> strategy.consolidate_knowledge()
        >>> len(set(strategy.get_new_clause_pairs()))
        9
        """

        support_set = self.support_set
        # noinspection PyTypeChecker
        return set(itertools.product(self._resolution_clauses - support_set, support_set)) | \
            set(itertools.combinations(support_set, 2))
