from abc import ABC
from typing import List, Iterable, Optional

from algorithms.abstract_search import SearchAlgorithmAttributes
from algorithms.blind_search import BlindSearchAlgorithm, ShortestPathSearch
from algorithms.helpers import InformedAlgorithmHelper
from data_structures.node import Node
from data_structures.treelikes import Treelike, Path


class InformedSearchAlgorithm(BlindSearchAlgorithm, ABC):
    _name = "Informed search algorithm"
    _short_description = "searches based on state space and heuristic"

    def __init__(self, attributes: SearchAlgorithmAttributes, helper: InformedAlgorithmHelper):
        super().__init__(attributes, helper)


class GreedyBestFirstSearch(InformedSearchAlgorithm):
    _name = "Greedy best first search"
    _short_description = "searches for the first state that is heuristically closest to the goal"

    def _insert_nodes(self,
                      treelike: Treelike,
                      legal_transition_nodes: Iterable[Node],
                      current_node: Node) -> Iterable[Node]:
        treelike.open.extend(legal_transition_nodes)
        treelike.open.sort(key=lambda node: node.state.heuristic)
        return legal_transition_nodes


class HillClimbingSearch(InformedSearchAlgorithm):
    _name = "Rise to the top search"
    _short_description = "searches for the first state the is heuristically closest to the goal discarding other states"
    _current_contender: Optional[Node]

    def _should_stop(self, treelike: Treelike, current_node) -> bool:
        treelike.open.extend(self._helper.get_legal_transition_nodes(current_node))

        return self._stuck(treelike, current_node) or self._found_end_node(treelike, current_node)

    def _stuck(self, treelike: Treelike, current_node: Node) -> bool:
        if len(treelike.open) == 0:
            # Should be a dead end, so in case it was the end node.
            if self._helper.found_end_node(current_node):
                treelike.path = Path(current_node.full_path)
            return True

        return False

    def _found_end_node(self, treelike: Treelike, current_node: Node) -> bool:
        self._current_contender = self._find_node_with_min_heuristic(treelike.open)

        if current_node.state.heuristic < self._current_contender.state.heuristic:
            # Should always pass, but just in case the heuristic is bad.
            if self._helper.found_end_node(current_node):
                treelike.path = Path(current_node.full_path)
            return True

        return False

    def _calculate_next_step(self, treelike: Treelike, current_node: Node) -> None:
        if self._current_contender is None:
            # Shouldn't be None, because this is executed after _found_end_node, but just in case
            self._current_contender = self._find_node_with_min_heuristic(treelike.open)
        inserted_nodes = self._insert_nodes(treelike, [self._current_contender], current_node)

        current_node.update_children(inserted_nodes)

    @staticmethod
    def _find_node_with_min_heuristic(nodes: List[Node]) -> Node:
        return sorted(nodes, key=lambda node: node.state.heuristic)[0]

    def _insert_nodes(self,
                      treelike: Treelike,
                      legal_transition_nodes: Iterable[Node],
                      current_node: Node) -> Iterable[Node]:

        treelike.open.clear()
        treelike.open.extend(legal_transition_nodes)
        return legal_transition_nodes


class AStarSearchAlgorithm(ShortestPathSearch, InformedSearchAlgorithm):
    _name = "A* search algorithm"
    _short_description = "takes everything into account :)"

    @staticmethod
    def _compare_key(node: Node):
        return node.cost + node.state.heuristic
