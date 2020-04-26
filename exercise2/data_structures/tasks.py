from abc import *

from data_structures.clause import Clause


class Task(ABC):
    operator: str = ''
    _clause: Clause

    @property
    def clause(self) -> Clause:
        return self._clause

    @staticmethod
    def parse(string_value: str) -> 'Task':
        try:
            normal_clause = Clause.parse(string_value[:-1])
            operator = string_value[-1]
        except IndexError:
            raise ValueError('Ill formed task string %s' % string_value)

        if operator == Inquiry.operator:
            return Inquiry(normal_clause)
        elif operator == PremiseAddition.operator:
            return PremiseAddition(normal_clause)
        elif operator == PremiseRemoval.operator:
            return PremiseRemoval(normal_clause)

        raise ValueError('Ill formed task string %s' % string_value)

    def __init__(self, normal_clause: Clause):
        self._clause = normal_clause

    def __str__(self) -> str:
        return '%s %s' % (str(self._clause), self.operator)

    def __repr__(self) -> str:
        return str(self)


class Inquiry(Task):
    operator: str = '?'


class PremiseAddition(Task):
    operator: str = '+'


class PremiseRemoval(Task):
    operator: str = '-'
