class Literal:
    @property
    def is_negated(self) -> bool:
        return self._is_negated

    def __init__(self, variable: str, is_negated=False):
        self._variable = variable
        self._is_negated = is_negated

    @staticmethod
    def parse(string_value: str):
        if string_value[0] == '~':
            is_negated = True
            variable = string_value[1:]
        else:
            is_negated = False
            variable = string_value

        return Literal(variable, is_negated)

    def __neg__(self) -> 'Literal':
        return Literal(self._variable, not self._is_negated)

    def __hash__(self) -> int:
        return ~hash(self._variable) if self._is_negated else hash(self._variable)

    def __ne__(self, other: 'Literal') -> bool:
        return not self == other

    def __eq__(self, other: 'Literal') -> bool:
        return self._variable.lower() == other._variable.lower() and self._is_negated == other._is_negated

    def __str__(self):
        return '~' if self._is_negated else '' + self._variable
