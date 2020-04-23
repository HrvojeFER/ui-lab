from data_structures.clause import NormalClause


class NormalCommand:
    @property
    def clause(self):
        return self._clause

    @staticmethod
    def parse(string_value: str):
        normal_clause = NormalClause.parse(string_value[:-1])
        operator = string_value[-1]

        if operator == NormalInquiry.operator:
            return NormalInquiry(normal_clause)
        elif operator == NormalClauseAddition.operator:
            return NormalClauseAddition(normal_clause)
        elif operator == NormalClauseErasure.operator:
            return NormalClauseErasure(normal_clause)

    def __init__(self, normal_clause: NormalClause):
        self._clause = normal_clause


class NormalInquiry(NormalCommand):
    operator = '?'


class NormalClauseAddition(NormalCommand):
    operator = '+'


class NormalClauseErasure(NormalCommand):
    operator = '-'
