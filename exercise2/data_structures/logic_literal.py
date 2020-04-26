from typing import *


class LogicLiteral(Hashable):
    _variable: str
    _is_negated: bool

    @property
    def is_negated(self) -> bool:
        return self._is_negated

    def __init__(self, variable: str, is_negated: bool = False):
        self._variable = variable
        self._is_negated = is_negated

    @staticmethod
    def parse(string_value: str):
        """
        >>> LogicLiteral.parse('a') == (LogicLiteral.parse('~a'))
        False
        """

        if string_value[0] == '~':
            is_negated = True
            variable = string_value[1:]
        else:
            is_negated = False
            variable = string_value

        return LogicLiteral(variable, is_negated)

    def is_complement(self, other: 'LogicLiteral') -> bool:
        return self.has_same_variable_as(other) and not self.has_same_polarity_as(other)

    def has_same_variable_as(self, other: 'LogicLiteral') -> bool:
        return self._variable.lower() == other._variable.lower()

    def has_same_polarity_as(self, other: 'LogicLiteral') -> bool:
        return self._is_negated == other._is_negated

    def __neg__(self) -> 'LogicLiteral':
        return LogicLiteral(self._variable, not self._is_negated)

    def __hash__(self) -> int:
        return ~hash(self._variable) if self._is_negated else hash(self._variable)

    def __ne__(self, other: 'LogicLiteral') -> bool:
        return not self == other

    def __eq__(self, other: 'LogicLiteral') -> bool:
        return self.has_same_variable_as(other) and self.has_same_polarity_as(other)

    def __str__(self) -> str:
        return ('~' if self._is_negated else '') + self._variable

    def __repr__(self) -> str:
        return str(self)
