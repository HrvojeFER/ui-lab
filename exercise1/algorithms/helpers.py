from abc import ABC, abstractmethod
from typing import Set

from input.classes import StateSpace, Goal, Heuristic
from input.file_manager import InputFileManager, InputKey
from data_structures.treelikes import Tree, Forest
from data_structures.node import State, Node
from data_structures.views.node import NodeView


class AlgorithmHelper(ABC):
    @property
    @abstractmethod
    def goal(self) -> Goal:
        pass

    def found_end_node(self, node: NodeView) -> bool:
        return node.state.name in self.goal.end_state_names

    @abstractmethod
    def _get_state(self, state_name: str) -> State:
        pass

    def create_search_tree(self) -> Tree:
        return Tree(self._get_state(self.goal.start_state_name))

    def create_search_forest(self) -> Forest:
        return Forest(self._get_state(self.goal.start_state_name))

    @abstractmethod
    def get_legal_transition_nodes(self, node: NodeView) -> Set[Node]:
        pass


class BlindAlgorithmHelper(AlgorithmHelper, ABC):
    @abstractmethod
    def _get_state(self, state_name: str) -> State:
        pass


class BlindDataManager(BlindAlgorithmHelper, InputFileManager):
    _state_space: StateSpace
    _goal: Goal

    def __init__(self, input_key: InputKey):
        state_space_lines = self.get_state_space_lines(input_key)
        state_space = StateSpace.parse(state_space_lines)

        search_goal_lines = self.get_search_goal_lines(input_key)
        search_goal = Goal.parse(search_goal_lines)

        self._state_space = state_space
        self._goal = search_goal

    @property
    def goal(self) -> Goal:
        return self._goal

    @goal.setter
    def goal(self, goal: Goal) -> None:
        self._goal = goal

    @property
    def state_space(self) -> StateSpace:
        return self._state_space

    def _get_state(self, state_name: str) -> State:
        return State(state_name)

    def get_legal_transition_nodes(self, node: Node) -> Set[Node]:
        legal_transition_nodes: Set[Node] = set()
        transitions = self._state_space[node.state.name]

        for connected_state_name in transitions:
            state = self._get_state(connected_state_name)
            cost = node.cost + transitions[connected_state_name]
            path = node.path + [node]

            new_node = Node(state, cost, path)
            legal_transition_nodes.add(new_node)

        return legal_transition_nodes


class InformedAlgorithmHelper(BlindAlgorithmHelper, ABC):
    @abstractmethod
    def _get_state(self, state_name: str) -> State:
        pass


class InformedDataManager(BlindDataManager, InformedAlgorithmHelper):
    _heuristic: Heuristic

    def __init__(self, input_key: InputKey):
        super().__init__(input_key)

        heuristic_lines = self.get_heuristics_lines(input_key)
        heuristic = Heuristic.parse(heuristic_lines)

        self._heuristic = heuristic

    @property
    def heuristic(self) -> Heuristic:
        return self._heuristic

    def _get_state(self, state_name: str) -> State:
        return State(state_name, self._heuristic[state_name])
