from abc import ABC
from itertools import count
from typing import Iterable, List, Optional, Hashable

from algorithms.abstract_search import AbstractSearchAlgorithm, SearchAlgorithmAttributes
from algorithms.helpers import BlindAlgorithmHelper
from data_structures.treelikes import Treelike
from data_structures.node import Node


class BlindSearchAlgorithm(AbstractSearchAlgorithm, ABC):
    _name = "Blind search algorithm"
    _short_description = "searches based on state space"

    def __init__(self, attributes: SearchAlgorithmAttributes, helper: BlindAlgorithmHelper):
        super().__init__(attributes, helper)


class BreadthFirstSearch(BlindSearchAlgorithm):
    _name = "Breadth first search"
    _short_description = "searches by levels"

    def _insert_nodes(self,
                      treelike: Treelike,
                      legal_transition_nodes: Iterable[Node],
                      current_node: Node) -> Iterable[Node]:

        inserted = set(legal_transition_nodes).difference(treelike.history + treelike.open)
        treelike.open.extend(inserted)
        return inserted


class UniformCostSearch(AbstractSearchAlgorithm):
    _name = "Uniform cost search"
    _short_description = "searches by cost"
    _open_trigger = 600
    _open_limit = 400
    _history_trigger = 2000
    _history_limit = 1000

    def _populate_treelike(self, treelike: Treelike) -> None:
        treelike.visit_count = 0
        super()._populate_treelike(treelike)

    @staticmethod
    def _compare_key(node: Node) -> int:
        return node.cost

    @staticmethod
    def _check_and_limit(trigger: int, limit: int, my_list: List):
        if len(my_list) > trigger:
            del my_list[limit:]

    def _insert_nodes(self,
                      treelike: Treelike,
                      legal_transition_nodes: Iterable[Node],
                      current_node: Node) -> Iterable[Node]:

        treelike.visit_count += 1
        self._check_and_limit(self._open_trigger, self._open_limit, treelike.open)
        self._check_and_limit(self._history_trigger, self._history_limit, treelike.history)

        to_extend = []
        replaced = []
        treelike.history.sort(key=self._compare_key)
        for node in legal_transition_nodes:
            open_index = None
            for index, open_node in enumerate(treelike.open):
                if hash(open_node.state.name) != hash(node.state.name):
                    continue
                if open_node.state.name == node.state.name:
                    open_index = index
                    break

            if open_index is not None:
                if self._compare_key(treelike.open[open_index]) > self._compare_key(node):
                    treelike.open[open_index] = node
                    replaced.append(node)
                continue

            found_in_history = False
            for history_node in treelike.history:
                if hash(history_node) != hash(node):
                    continue
                if history_node.state.name == node.state.name:
                    found_in_history = True
                    break
            if found_in_history:
                continue

            to_extend.append(node)

        treelike.open.extend(to_extend)
        treelike.open.sort(key=self._compare_key)
        return replaced + to_extend


class DepthFirstSearch(BlindSearchAlgorithm):
    _name = "Depth first search"
    _short_description = "searches in depth"

    def _insert_nodes(self,
                      treelike: Treelike,
                      legal_transition_nodes: Iterable[Node],
                      current_node: Node) -> Iterable[Node]:

        result_open = list(legal_transition_nodes) + treelike.open
        treelike.open.clear()
        treelike.open.extend(result_open)
        return legal_transition_nodes


class LimitedDepthFirstSearch(DepthFirstSearch):
    _name = "Limited depth first search"
    _short_description = "searches in depth until a certain limit"
    _default_depth_limit = 5

    def __init__(self, attributes: SearchAlgorithmAttributes, helper: BlindAlgorithmHelper):
        if attributes.depth_limit is None:
            attributes.depth_limit = self._default_depth_limit
        super().__init__(attributes, helper)

    def _insert_nodes(self,
                      treelike: Treelike,
                      legal_transition_nodes: Iterable[Node],
                      current_node: Node) -> Iterable[Node]:

        if current_node.depth < self.attributes.depth_limit:
            return super()._insert_nodes(treelike, legal_transition_nodes, current_node)

        return []


class IterativeDepthFirstSearch(LimitedDepthFirstSearch):
    _name = "Limited depth first search"
    _short_description = "repeatedly searches in depth until a certain limit with a higher limit every time"

    @property
    def current_depth_limit(self) -> int:
        return self.attributes.depth_limit

    @current_depth_limit.setter
    def current_depth_limit(self, limit: int) -> None:
        self.attributes.depth_limit = limit

    def _generate_treelike(self) -> Treelike:
        search_forest = self._helper.create_search_forest()
        self._populate_treelike(search_forest)
        return search_forest

    def _populate_treelike(self, treelike: Treelike) -> None:
        for i in count():
            self.current_depth_limit = i
            super()._step_over_nodes(treelike)

            if treelike.path_found:
                break

            treelike.reset()


class ShortestPathSearch(UniformCostSearch):
    _name = "Shortest path"
    _short_description = "finds the definite shortest path to the end node"
