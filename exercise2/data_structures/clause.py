from typing import *

from data_structures.logic_literal import LogicLiteral


class Clause(FrozenSet[LogicLiteral], Hashable):
    disjunction = 'v'
    NIL = 'NIL'

    @staticmethod
    def parse(string_value: str) -> 'Clause':
        """
        >>> Clause.parse('a v b') == Clause.parse('~a v b')
        False
        """

        logic_literals = set()
        for element in string_value.split():
            if element.lower() != Clause.disjunction:
                logic_literals.add(LogicLiteral.parse(element))

        return Clause(logic_literals)

    @staticmethod
    def remove_complementary_literals(logic_literals: Iterable[LogicLiteral]) -> FrozenSet[LogicLiteral]:
        logic_literal_set: Set[LogicLiteral] = set(logic_literals)

        for logic_literal in logic_literals:
            complement = -logic_literal
            if complement in logic_literal_set:
                logic_literal_set -= {logic_literal, complement}

        return frozenset(logic_literal_set)

    def is_tautology(self) -> bool:
        for logic_literal in self:
            if -logic_literal in self:
                return True

        return False

    def is_nil(self) -> bool:
        return len(self) == 0

    def __neg__(self) -> FrozenSet['Clause']:
        return frozenset(Clause({-logic_literal}) for logic_literal in self)

    def __hash__(self) -> int:
        return sum((hash(literal) for literal in self))

    def __str__(self) -> str:
        if not self.is_nil():
            return (' %s ' % self.disjunction).join([str(logic_literal)
                                                     for logic_literal in sorted(self, key=lambda x: str(x))])
        return Clause.NIL

    def __repr__(self) -> str:
        return str(self)
