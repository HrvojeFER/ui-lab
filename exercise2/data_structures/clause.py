from typing import Iterable, Iterator
from data_structures.literal import Literal


class NormalClause(Iterable[Literal]):
    disjunction = 'v'

    def __init__(self, literals: Iterable[Literal]):
        self._literals = set(literals)

    @staticmethod
    def parse(string_value: str):
        literals = set()
        for element in string_value.split():
            if element != NormalClause.disjunction:
                literals.add(Literal(element))

        return NormalClause(literals)

    def is_nil(self) -> bool:
        return len(self._literals) == 0

    def __iter__(self) -> Iterator[Literal]:
        return iter(self._literals)

    def __ne__(self, other: 'NormalClause') -> bool:
        return not self == other

    def __eq__(self, other: 'NormalClause') -> bool:
        return self._literals == other._literals

    def __hash__(self) -> int:
        return sum((hash(literal) for literal in self._literals))

    def __str__(self) -> str:
        return ' '.join([str(element) for element in self._literals])
