from typing import Set, Iterable, List, Optional

from data import State, StateSpace, Heuristic


class SearchTreeNode(Iterable):
    _state: State
    _depth: int
    _cost: int
    children: Set
    path: List

    def __init__(self, state: State, depth: int, cost: int, path: List = None):
        self._state = state
        self._depth = depth
        self._cost = cost
        self.children = set()
        self.path = [] if path is None else path

    def __iadd__(self, other: Iterable):
        self.children.update(other)
        return self

    def __iter__(self):
        return iter(self.children)

    def __len__(self):
        return len(self.children)

    def get_state(self) -> State:
        return self._state

    def get_depth(self) -> int:
        return self._depth

    def get_cost(self) -> int:
        return self._cost

    def stringify_recursive_with_path_highlight(self, highlight: List[str]) -> List[str]:
        result = []

        for child in self:
            result += child.stringify_recursive_with_path_highlight(highlight)

        # TODO: fix
        if self._state in highlight:
            self_str = '\033[92m' + str(self) + '\033[0m'
        else:
            self_str = str(self)

        result.append(self._depth * '\t' + self_str)

        return result

    def __str__(self):
        return str(self._state) + ': ' + str(self._cost)


class SearchPath(Iterable[SearchTreeNode]):
    _nodes: List[SearchTreeNode]

    def __init__(self, nodes: List[SearchTreeNode]):
        self._nodes = nodes

    def get_cost(self) -> int:
        return self._nodes[-1].get_cost()

    def __len__(self) -> int:
        return len(self._nodes)

    def __getitem__(self, item: int) -> SearchTreeNode:
        return self._nodes[item]

    def __iadd__(self, other: SearchTreeNode):
        self._nodes.append(other)
        return self

    def __iter__(self):
        return iter(self._nodes)

    def __str__(self):
        result = ''

        for state in self._nodes:
            result += str(state) + ' =>\n'

        return result[:-4]


class SearchTree:
    _root: SearchTreeNode
    _closed: List[SearchTreeNode]
    _open: List[SearchTreeNode]
    path: Optional[SearchPath]

    def __init__(self, root_state: State):
        self._root = SearchTreeNode(root_state, 0, 0)
        self._closed = []
        self._open = [self._root]
        self.path = None

    def get_open(self) -> List[SearchTreeNode]:
        return self._open

    def get_visited(self) -> List[SearchTreeNode]:
        return self._open + self._closed

    def reset(self) -> None:
        self._closed += self._open
        self._open = [self._root]

    def step(self) -> SearchTreeNode:
        step = self._open[0]
        self._open = self._open[1:]

        self._closed += step
        return step

    def path_found(self) -> bool:
        return self.path is not None

    def __str__(self):
        highlight = [node.get_state().get_name() for node in self.path]
        lines = self._root.stringify_recursive_with_path_highlight(highlight)
        lines.reverse()
        return '\n'.join(lines) + '\n'


class SearchDataManager:
    _state_space: StateSpace
    _heuristic: Optional[Heuristic]

    def __init__(self, state_space: StateSpace, heuristic: Optional[Heuristic]):
        self._state_space = state_space
        self._heuristic = heuristic

    def _get_heuristic(self, state_name: str) -> Optional[int]:
        return None if self._heuristic is None else self._heuristic[state_name]

    def get_state(self, state_name: str) -> State:
        return State(state_name, self._get_heuristic(state_name))

    def get_legal_transition_nodes(self, node: SearchTreeNode) -> Set[SearchTreeNode]:
        legal_transition_nodes: Set[SearchTreeNode] = set()
        transitions = self._state_space[node.get_state().get_name()]

        for connected_state_name in transitions:
            state = self.get_state(connected_state_name)
            depth = node.get_depth() + 1
            cost = node.get_cost() + transitions[connected_state_name]
            path = node.path + [node]

            new_node = SearchTreeNode(state, depth, cost, path)
            legal_transition_nodes.add(new_node)

        return legal_transition_nodes
