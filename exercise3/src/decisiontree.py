from __future__ import annotations

from typing import *


class Leaf:
    def __init__(self, class_: str):
        self._class: str = class_

    @property
    def class_(self) -> str:
        return self._class

    def __str__(self) -> str:
        return self._class


class Node:
    def __init__(self, feature: str, children: Iterable[Tuple[str, Union[Node, Leaf]]]):
        self._feature: str = feature
        self._children: Set[Tuple[str, Union[Node, Leaf]]] = set(children)

    def iterate(self, feature_values: Iterable[str]) -> str:
        feature_values = set(feature_values)
        for child in self._children:
            if child[0] in feature_values:
                if isinstance(child[1], Node):
                    feature_values = list(feature_values)
                    feature_values.remove(child[0])
                    return child[1].iterate(feature_values)
                else:
                    return child[1].class_
