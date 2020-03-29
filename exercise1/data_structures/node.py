from itertools import chain
from typing import Optional, List, Iterator, Set, Iterable, Hashable
from dataclasses import dataclass
from data_structures.views.node import NodeView, StateView
from data_structures.iteration_level import IterationLevel


class Node(NodeView, Hashable):
    _state: 'State'
    _cost: int
    _children: Set['Node']
    _path: List['Node']

    def __init__(self, state: 'State', cost: int, path: List['Node'] = None):
        self._state = state
        self._cost = cost
        self._children = set()
        self._path = [] if path is None else path

    @property
    def children(self) -> Set['Node']:
        return self._children

    def update_children(self, other_children: Iterable['Node']) -> None:
        self._children.update(other_children)

    @property
    def path(self) -> List['Node']:
        return self._path

    @path.setter
    def path(self, path: List['Node']) -> None:
        self._path = path

    @property
    def full_path(self) -> List['Node']:
        return self.path + [self]

    @property
    def state(self) -> 'State':
        return self._state

    @property
    def cost(self) -> int:
        return self._cost

    def iterate_over(self, iteration_level: IterationLevel) -> Iterator['Node']:
        result = []

        if iteration_level == IterationLevel.roots:
            result = [self]
        elif iteration_level == IterationLevel.branches:
            result = reversed(list(self._iterate_over_branches()))
        elif iteration_level == IterationLevel.leaves:
            result = reversed(list(self._iterate_over_leaves()))

        return iter(result)

    def _iterate_over_branches(self) -> Iterator['Node']:
        if self.has_children:
            for child in self:
                yield from child._iterate_over_branches()

        yield self

    def _iterate_over_leaves(self) -> Iterator['Node']:
        for child in self:
            yield from child._iterate_over_leaves()

        yield self

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Node):
            return self.state.name == o.state.name
        return False

    def __hash__(self) -> int:
        return hash(self.state.name)

    def __iter__(self) -> Iterator['Node']:
        return iter(self.children)


@dataclass
class State(StateView):
    _name: str
    _heuristic: Optional[int] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def heuristic(self) -> Optional[int]:
        return self._heuristic

    @heuristic.setter
    def heuristic(self, heuristic: int) -> None:
        self._heuristic = heuristic
