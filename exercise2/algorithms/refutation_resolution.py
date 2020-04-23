from abc import *
from typing import Optional, Iterable, Generator, Set, Tuple, List
import itertools
from data_structures.clause import NormalClause
from data_structures.literal import Literal


class RefutationResolution:
    @property
    def clauses(self) -> Set[NormalClause]:
        return self._control_strategy.clauses

    @property
    def goal(self) -> NormalClause:
        return self._control_strategy.goal

    @goal.setter
    def goal(self, new_goal: NormalClause) -> None:
        self._control_strategy.goal = new_goal

    def __init__(self, control_strategy: 'ControlStrategy', simplification_strategy: 'SimplificationStrategy'):
        self._control_strategy = control_strategy
        self._simplification_strategy = simplification_strategy

    @staticmethod
    def resolve(clause_a: NormalClause, clause_b: NormalClause) -> NormalClause:
        result_literals: List[Literal] = []

        literal_pair: Tuple[Literal, Literal]
        for literal_pair in itertools.product(clause_a, clause_b):
            if literal_pair[0] != -literal_pair[1]:
                result_literals.extend(literal_pair)

        return NormalClause(result_literals)

    def resolve_goal(self, goal: Optional[NormalClause] = None) -> bool:
        self._control_strategy.prepare_for_resolution(goal)

        while True:
            for clause_a, clause_b in self._control_strategy.get_new_clause_pairs():
                resolvent = self.resolve(clause_a, clause_b)

                if resolvent.is_nil():
                    return True

                self._control_strategy.update_new_clauses(resolvent)

            if not self._control_strategy.found_anything_new():
                return False

            self._control_strategy.update_resolution_clauses()


class ControlStrategy(ABC):
    @property
    def clauses(self) -> Set[NormalClause]:
        return self._clauses

    @property
    def goal(self) -> NormalClause:
        return self._goal

    @goal.setter
    def goal(self, new_goal: NormalClause) -> None:
        self._goal = new_goal

    def __init__(self, clauses: Iterable[NormalClause], goal: NormalClause):
        self._clauses: Set[NormalClause] = set(clauses)
        self._goal: NormalClause = goal

        self._resolution_clauses: Set[NormalClause] = set()
        self._new_clauses: Set[NormalClause] = set()

    def prepare_for_resolution(self, goal: Optional[NormalClause] = None) -> None:
        if goal is not None:
            self._goal = goal

        self._resolution_clauses = self._clauses
        self._new_clauses = set()

    def update_resolution_clauses(self) -> None:
        self._resolution_clauses.update(self._new_clauses)

    def update_new_clauses(self, clauses: Iterable[NormalClause]) -> None:
        self._new_clauses.update(clauses)

    def found_anything_new(self) -> bool:
        return self._new_clauses.issubset(self._clauses)

    @abstractmethod
    def get_new_clause_pairs(self) -> Set[Tuple[NormalClause, NormalClause]]:
        pass


class SaturationByLevels(ControlStrategy):
    pass


class SupportSetStrategy(ControlStrategy):
    pass


class SimplificationStrategy(ABC):
    @abstractmethod
    def simplify(self, clauses: Iterable[NormalClause]) -> Generator[NormalClause]:
        pass


class RedundantClauseRemoval(SimplificationStrategy):
    pass


class InsignificantClauseRemoval(SimplificationStrategy):
    pass

