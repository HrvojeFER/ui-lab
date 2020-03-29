from abc import ABC, abstractmethod
from typing import Hashable, Optional, Collection, List, Set, Iterator
from data_structures.iteration_level import IterationLevel


class NodeView(Collection, Hashable, ABC):
    @property
    def has_children(self) -> bool:
        return len(self.children) > 0

    @property
    @abstractmethod
    def children(self) -> Set['NodeView']:
        pass

    @property
    @abstractmethod
    def path(self) -> List['NodeView']:
        pass

    @property
    @abstractmethod
    def full_path(self) -> List['NodeView']:
        pass

    @property
    def depth(self) -> int:
        return len(self.path)

    @property
    @abstractmethod
    def state(self) -> 'StateView':
        pass

    @property
    @abstractmethod
    def cost(self) -> int:
        pass

    @abstractmethod
    def iterate_over(self, iteration_key: IterationLevel) -> Iterator['NodeView']:
        pass

    def __len__(self) -> int:
        return self.depth

    def __contains__(self, obj: object) -> bool:
        return obj in set(iter(self))

    def __str__(self):
        return str(self.state) + ' -> ' + str(self.cost)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, NodeView):
            return self.state.name == o.state.name
        return False

    def __hash__(self) -> int:
        return hash(self.state.name)


class StateView(Hashable, ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def heuristic(self) -> Optional[int]:
        pass

    @property
    def has_heuristic(self) -> bool:
        return self.heuristic is not None

    def __str__(self) -> str:
        return self.name + ((': ' + str(self.heuristic)) if self.has_heuristic else '')

    def __eq__(self, o: object) -> bool:
        if isinstance(o, StateView):
            return o.name == self.name

        return False

    def __hash__(self) -> int:
        return hash(self.name)
