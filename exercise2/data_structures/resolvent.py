from data_structures.clause import Clause


class Resolvent(Clause):
    _left_parent: Clause
    _right_parent: Clause

    @property
    def left_parent(self) -> Clause:
        return self._left_parent

    @property
    def right_parent(self) -> Clause:
        return self._right_parent

    def set_parents(self, left_parent: Clause, right_parent: Clause) -> 'Resolvent':
        self._left_parent = left_parent
        self._right_parent = right_parent

        return self

    def is_new_knowledge(self) -> bool:
        return self < self._left_parent | self._right_parent
