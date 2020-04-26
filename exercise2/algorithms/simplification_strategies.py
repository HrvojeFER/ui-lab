from abc import *
from typing import *
import itertools

from data_structures.clause import Clause


class SimplificationStrategy(ABC):
    @abstractmethod
    def simplify(self, clauses: Iterable[Clause]) -> Iterable[Clause]: ...


class RedundantClauseRemoval(SimplificationStrategy):
    def simplify(self, clauses: Iterable[Clause]) -> Set[Clause]:
        """
        >>> str(RedundantClauseRemoval().simplify( \
                {Clause.parse('f v ~c'), Clause.parse('f v g v ~c'), Clause.parse('f')}))
        '{f}'
        """

        to_remove: Set[Clause] = set()
        for clause_a, clause_b in itertools.combinations(clauses, 2):
            if clause_a <= clause_b:
                to_remove.add(clause_b)
            elif clause_b <= clause_a:
                to_remove.add(clause_a)

        if isinstance(clauses, set):
            return clauses - to_remove
        else:
            return set(clauses) - to_remove


class InsignificantClauseRemoval(SimplificationStrategy):
    def simplify(self, clauses: Iterable[Clause]) -> Generator[Clause, None, None]:
        """
        >>> str(set(InsignificantClauseRemoval().simplify({Clause.parse('f v g v ~f')})))
        'set()'
        """

        for clause in clauses:
            if not clause.is_tautology():
                yield clause
