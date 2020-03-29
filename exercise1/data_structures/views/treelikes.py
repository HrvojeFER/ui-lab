from abc import ABC, abstractmethod
from typing import Sequence, List, Optional, Iterator, Iterable

from data_structures.iteration_level import IterationLevel
from data_structures.views.node import NodeView


class PathView(ABC, Sequence[NodeView]):
    @property
    @abstractmethod
    def cost(self) -> int:
        pass

    def __str__(self) -> str:
        result = ''

        for node in self:
            result += str(node) + ' =>\n'

        return result[:-4]


class TreelikeView(Iterable[NodeView], ABC):
    @property
    @abstractmethod
    def root(self) -> NodeView:
        pass

    @property
    def path_found(self) -> bool:
        return self.path is not None

    @property
    @abstractmethod
    def path(self) -> Optional[PathView]:
        pass

    @property
    @abstractmethod
    def visited(self) -> List[NodeView]:
        pass

    @property
    @abstractmethod
    def visit_count(self) -> int:
        pass

    @abstractmethod
    def iterate_over(self, iteration_level: IterationLevel) -> Iterator[NodeView]:
        pass

    def stringify(self, iteration_level: IterationLevel) -> str:
        result = ''

        for node in self.iterate_over(iteration_level):
            result += node.depth * '\t' + str(node) + '\n'

        return result + '\n'

    def __str__(self):
        self.stringify(IterationLevel.leaves)
