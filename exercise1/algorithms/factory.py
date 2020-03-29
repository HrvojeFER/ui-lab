from enum import Enum
from typing import Union

from algorithms.abstract_search import AbstractSearchAlgorithm, SearchAlgorithmAttributes
from algorithms.helpers import InformedAlgorithmHelper, BlindAlgorithmHelper

from algorithms.blind_search import \
    BreadthFirstSearch, \
    UniformCostSearch, \
    DepthFirstSearch, \
    LimitedDepthFirstSearch, \
    IterativeDepthFirstSearch, \
    ShortestPathSearch

from algorithms.informed_search import \
    GreedyBestFirstSearch, \
    HillClimbingSearch, \
    AStarSearchAlgorithm


class BlindSearchAlgorithmKey(Enum):
    BreadthFirstSearch = 0,
    UniformCostSearch = 1,
    DepthFirstSearch = 2,
    LimitedDepthFirstSearch = 3,
    IterativeDepthFirstSearch = 4,
    ShortestPathSearch = 5


class InformedSearchAlgorithmKey(Enum):
    GreedyBestFirstSearch = 5,
    HillClimbingSearch = 6,
    AStarSearchAlgorithm = 7


class SearchAlgorithmFactory:
    @classmethod
    def create_algorithm(cls,
                         key: Union[BlindSearchAlgorithmKey, InformedSearchAlgorithmKey],
                         helper: InformedAlgorithmHelper,
                         attributes: SearchAlgorithmAttributes) -> AbstractSearchAlgorithm:

        if isinstance(key, BlindSearchAlgorithmKey):
            return cls.create_blind_algorithm(key, helper, attributes)
        else:
            return cls.create_informed_algorithm(key, helper, attributes)

    @staticmethod
    def create_blind_algorithm(key: BlindSearchAlgorithmKey,
                               helper: BlindAlgorithmHelper,
                               attributes: SearchAlgorithmAttributes) -> AbstractSearchAlgorithm:

        if key == BlindSearchAlgorithmKey.BreadthFirstSearch:
            return BreadthFirstSearch(attributes, helper)
        if key == BlindSearchAlgorithmKey.UniformCostSearch:
            return UniformCostSearch(attributes, helper)
        if key == BlindSearchAlgorithmKey.DepthFirstSearch:
            return DepthFirstSearch(attributes, helper)
        if key == BlindSearchAlgorithmKey.LimitedDepthFirstSearch:
            return LimitedDepthFirstSearch(attributes, helper)
        if key == BlindSearchAlgorithmKey.IterativeDepthFirstSearch:
            return IterativeDepthFirstSearch(attributes, helper)
        if key == BlindSearchAlgorithmKey.ShortestPathSearch:
            return ShortestPathSearch(attributes, helper)

    @staticmethod
    def create_informed_algorithm(key: InformedSearchAlgorithmKey,
                                  helper: InformedAlgorithmHelper,
                                  attributes: SearchAlgorithmAttributes) -> AbstractSearchAlgorithm:
        if key == InformedSearchAlgorithmKey.GreedyBestFirstSearch:
            return GreedyBestFirstSearch(attributes, helper)
        if key == InformedSearchAlgorithmKey.HillClimbingSearch:
            return HillClimbingSearch(attributes, helper)
        if key == InformedSearchAlgorithmKey.AStarSearchAlgorithm:
            return AStarSearchAlgorithm(attributes, helper)
