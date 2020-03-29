from dataclasses import dataclass
from random import shuffle
from typing import Optional, List, Iterable
from abc import ABC, abstractmethod
from copy import deepcopy

from algorithms.helpers import AlgorithmHelper
from data_structures.treelikes import Treelike, Path
from data_structures.node import Node
from data_structures.views.treelikes import TreelikeView


@dataclass
class SearchAlgorithmAttributes:
    depth_limit: Optional[int] = None
    breadth_limit: Optional[int] = None
    chaotic: bool = False

    def __str__(self):
        return 'Limits: ' + str(self.depth_limit) + ' ' + str(self.breadth_limit) + '; Chaotic: ' + str(self.chaotic)


class AbstractSearchAlgorithm(ABC):
    _name: str = 'Abstract search algorithm'
    _short_description: str = 'base for all search algorithms'
    _attributes: SearchAlgorithmAttributes
    _my_helper: AlgorithmHelper

    def __init__(self, attributes: SearchAlgorithmAttributes, helper: AlgorithmHelper):
        self._attributes = deepcopy(attributes)
        self._my_helper = helper

    @property
    def attributes(self) -> SearchAlgorithmAttributes:
        return self._attributes

    @property
    def _helper(self) -> AlgorithmHelper:
        return self._my_helper

    def search(self) -> TreelikeView:
        return self._generate_treelike()

    def _generate_treelike(self) -> Treelike:
        search_tree = self._helper.create_search_tree()
        self._populate_treelike(search_tree)
        return search_tree

    def _populate_treelike(self, treelike: Treelike) -> None:
        self._step_over_nodes(treelike)

    def _step_over_nodes(self, treelike: Treelike) -> None:
        while len(treelike.open) != 0:
            current_node = treelike.step()
            if self._should_stop(treelike, current_node):
                break

            self._calculate_next_step(treelike, current_node)

    def _should_stop(self, treelike: Treelike, current_node) -> bool:
        if self._helper.found_end_node(current_node):
            treelike.path = Path(current_node.full_path)
            return True

        return False

    def _calculate_next_step(self, treelike: Treelike, current_node: Node) -> None:
        legal_transition_nodes = self._get_legal_transition_nodes(current_node)
        inserted_nodes = self._insert_nodes(treelike, legal_transition_nodes, current_node)
        current_node.update_children(inserted_nodes)

    def _get_legal_transition_nodes(self, current_node: Node) -> Iterable[Node]:
        nodes = self._helper.get_legal_transition_nodes(current_node)
        if self.attributes.chaotic:
            nodes = list(nodes)
            shuffle(nodes)
        return nodes

    @abstractmethod
    def _insert_nodes(self,
                      treelike: Treelike,
                      legal_transition_nodes: Iterable[Node],
                      current_node: Node) \
            -> Iterable[Node]:
        pass

    def __str__(self):
        return self._name + ': ' + self._short_description + '; ' + str(self.attributes)
