from abc import ABC, abstractmethod
from typing import Sequence, List, Iterator, Optional, Collection, Iterable
from itertools import chain

from data_structures.iteration_level import IterationLevel
from data_structures.node import State, Node
from data_structures.views.treelikes import PathView, TreelikeView


class Path(PathView, Sequence[Node]):
    _nodes: List[Node]

    def __init__(self, nodes: List[Node]):
        self._nodes = nodes

    @property
    def cost(self) -> int:
        return self._nodes[-1].cost

    def __getitem__(self, item: int) -> Node:
        return self._nodes[item]

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self._nodes)

    def __iadd__(self, other: Node):
        self._nodes.append(other)
        return self


class Treelike(TreelikeView, ABC):
    @property
    @abstractmethod
    def root(self) -> Node:
        pass

    @property
    @abstractmethod
    def path(self) -> Optional[Path]:
        pass

    @path.setter
    @abstractmethod
    def path(self, path: Path) -> None:
        pass

    @property
    @abstractmethod
    def open(self) -> List[Node]:
        pass

    @property
    def visited(self) -> List[Node]:
        return self.history

    _visit_count = None

    @property
    def visit_count(self) -> int:
        return self._visit_count

    @visit_count.setter
    def visit_count(self, value: int) -> None:
        self._visit_count = value

    @property
    @abstractmethod
    def history(self) -> List[Node]:
        pass

    @abstractmethod
    def step(self) -> Node:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def iterate_over(self, iteration_level: IterationLevel) -> Iterator[Node]:
        pass

    def __iter__(self) -> Iterator[Node]:
        return self.iterate_over(IterationLevel.leaves)


class Tree(Treelike, Collection[Node]):
    _root: Node
    _open: List[Node]
    _closed: List[Node]
    _path: Optional[Path]

    def __init__(self, root_state: State):
        self._root = Node(root_state, cost=0)
        self._open = [self._root]
        self._closed = []
        self._path = None

    @property
    def root(self):
        return self._root

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @path.setter
    def path(self, path: Path) -> None:
        self._path = path

    @property
    def open(self) -> List[Node]:
        return self._open

    @property
    def history(self) -> List[Node]:
        return self._closed

    def step(self) -> Node:
        step = self._open[0]
        self._open = self._open[1:]

        self._closed.append(step)
        return step

    def reset(self) -> None:
        self.__init__(self._root.state)

    def __len__(self) -> int:
        return self.root.depth

    def __contains__(self, o: object):
        return o in set(iter(self))

    def iterate_over(self, iteration_level: IterationLevel) -> Iterator[Node]:
        if iteration_level == IterationLevel.visited:
            return (node for node in self.root.iterate_over(iteration_level.leaves) if
                    (id(node) in (id(visit) for visit in self.visited)))

        return self.root.iterate_over(iteration_level)


class Forest(Treelike, Collection[Tree]):
    _trees: List[Tree]

    def __init__(self, root_state: State):
        self._trees = [Tree(root_state)]

    @property
    def root(self) -> Node:
        return self._current_tree.root

    @property
    def path(self) -> Optional[Path]:
        return self._current_tree.path

    @path.setter
    def path(self, path: Path) -> None:
        self._current_tree.path = path

    @property
    def open(self) -> List[Node]:
        return self._current_tree.open

    @property
    def visited(self) -> List[Node]:
        return list(chain.from_iterable([tree.visited for tree in self._trees]))

    @property
    def history(self) -> List[Node]:
        return self._current_tree.history

    def step(self) -> Node:
        return self._current_tree.step()

    def reset(self) -> None:
        self._trees.append(Tree(self._current_tree.root.state))

    def __len__(self) -> int:
        return len(self._trees)

    def __contains__(self, o: object):
        return o in set(iter(self))

    def iterate_over(self, iteration_level: IterationLevel) -> Iterator[Node]:
        if iteration_level.value <= IterationLevel.roots.value:
            return iter(self)

        return chain.from_iterable([tree.iterate_over(iteration_level) for tree in self])

    @property
    def _current_tree(self):
        return self._trees[-1]
