from abc import ABC, abstractmethod
from itertools import count
from enum import Enum

from data import SearchGoal
from search_data import *


class AbstractSearchAlgorithm(ABC):
    _search_data_manager: SearchDataManager
    _chaotic: bool

    def __init__(self, search_data_manager: SearchDataManager, chaotic: bool = False):
        self._search_data_manager = search_data_manager
        self._chaotic = chaotic

    @abstractmethod
    def _insert_nodes(self,
                      search_tree: SearchTree,
                      nodes: Iterable[SearchTreeNode],
                      current_node: SearchTreeNode) \
            -> Iterable[SearchTreeNode]:
        pass

    def _search(self, goal: SearchGoal, search_tree: SearchTree) -> None:
        while len(search_tree.get_open()) != 0:
            current_node = search_tree.step()

            if current_node.get_state().get_name() in goal.end_state_names:
                search_tree.path = SearchPath(current_node.path + [current_node])
                break

            legal_transition_nodes = self._search_data_manager.get_legal_transition_nodes(current_node)
            if not self._chaotic:
                legal_transition_nodes = sorted(legal_transition_nodes, key=lambda node: str(node))

            current_node.children = self._insert_nodes(
                search_tree,
                legal_transition_nodes,
                current_node)

    def generate_search_tree(self, goal: SearchGoal) -> SearchTree:
        search_tree = SearchTree(self._search_data_manager.get_state(goal.start_state_name))

        self._search(goal, search_tree)

        return search_tree


class BreadthFirstSearch(AbstractSearchAlgorithm):
    def _insert_nodes(self,
                      search_tree: SearchTree,
                      nodes: Iterable[SearchTreeNode],
                      current_node: SearchTreeNode) \
            -> Iterable[SearchTreeNode]:

        search_tree.get_open().extend(nodes)
        return nodes


class CostEqualitySearch(AbstractSearchAlgorithm):
    def _insert_nodes(self,
                      search_tree: SearchTree,
                      nodes: Iterable[SearchTreeNode],
                      current_node: SearchTreeNode) \
            -> Iterable[SearchTreeNode]:

        search_tree.get_open().extend(nodes)
        search_tree.get_open().sort(key=lambda node: node.get_cost())
        return nodes


class DepthFirstSearch(AbstractSearchAlgorithm):
    def _insert_nodes(self,
                      search_tree: SearchTree,
                      nodes: Iterable[SearchTreeNode],
                      current_node: SearchTreeNode) \
            -> Iterable[SearchTreeNode]:

        result = list(nodes) + search_tree.get_open()
        search_tree.get_open().clear()
        search_tree.get_open().extend(result)
        return nodes


class LimitedDepthFirstSearch(DepthFirstSearch):
    _limit: int

    def __init__(self, data_manager: SearchDataManager, limit: int = 5, chaotic: bool = False):
        super().__init__(data_manager, chaotic)
        self._limit = limit

    def _insert_nodes(self,
                      search_tree: SearchTree,
                      nodes: Iterable[SearchTreeNode],
                      current_node: SearchTreeNode) \
            -> Iterable[SearchTreeNode]:

        if current_node.get_depth() < self._limit:
            return super()._insert_nodes(search_tree, nodes, current_node)

        return []


class IterativeDepthFirstSearch(LimitedDepthFirstSearch):
    def __init__(self, data_manager: SearchDataManager, chaotic: bool = False):
        super().__init__(data_manager, 1, chaotic)

    def _search(self, goal: SearchGoal, search_tree: SearchTree) -> None:
        for i in count():
            self._limit = i
            super()._search(goal, search_tree)

            if search_tree.path_found():
                break

            search_tree.reset()


class AlgorithmKeys(Enum):
    BreadthFirstSearch = 0,
    CostEqualitySearch = 1,
    DepthFirstSearch = 2,
    LimitedDepthFirstSearch = 3,
    IterativeDepthFirstSearch = 4


class AlgorithmFactory:
    _limit: int
    _chaotic: bool

    def __init__(self, limit: int = 5, chaotic: bool = False):
        self._limit = limit
        self._chaotic = chaotic

    def create_algorithm(self,
                         algorithm_key: AlgorithmKeys,
                         search_data_manager: SearchDataManager) -> AbstractSearchAlgorithm:

        if algorithm_key == AlgorithmKeys.BreadthFirstSearch:
            return BreadthFirstSearch(search_data_manager, self._chaotic)
        if algorithm_key == AlgorithmKeys.DepthFirstSearch:
            return DepthFirstSearch(search_data_manager, self._chaotic)
        if algorithm_key == AlgorithmKeys.LimitedDepthFirstSearch:
            return LimitedDepthFirstSearch(search_data_manager, self._limit, self._chaotic)
        if algorithm_key == AlgorithmKeys.CostEqualitySearch:
            return CostEqualitySearch(search_data_manager, self._chaotic)
        if algorithm_key == AlgorithmKeys.IterativeDepthFirstSearch:
            return IterativeDepthFirstSearch(search_data_manager, self._chaotic)
