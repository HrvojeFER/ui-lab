from dataclasses import dataclass
from typing import List, Hashable
from copy import deepcopy

from algorithms.abstract_search import AbstractSearchAlgorithm
from input.classes import Goal
from algorithms.helpers import InformedDataManager
from algorithms.factory import BlindSearchAlgorithmKey, SearchAlgorithmFactory, SearchAlgorithmAttributes
from data_structures.node import State
from data_structures.views.node import NodeView, StateView


@dataclass(frozen=True)
class OptimismError(Hashable):
    node: NodeView

    def __str__(self):
        return 'Node heuristic: ' + str(self.node.state) + '\t' + \
               'Expected max heuristic:' + str(self.node.cost)

    def __hash__(self) -> int:
        return hash(self.node.state.name)

    def __eq__(self, other):
        if isinstance(other, OptimismError):
            return self.node.state.name == other.node.state.name
        return False


@dataclass(frozen=True)
class ConsistencyError:
    state: StateView
    connected: StateView
    transition_cost: int

    def __str__(self):
        return 'Heuristic of: ' + str(self.state) + \
               ' is greater than heuristic of: ' + str(self.connected) + \
               ' + the transition cost: ' + str(self.transition_cost)


class HeuristicQualityCheckResult:
    _optimism_errors = List[OptimismError]
    _consistency_errors = List[ConsistencyError]

    def __init__(self):
        self._optimism_errors = []
        self._consistency_errors = []

    @property
    def optimism_errors(self) -> List[OptimismError]:
        return self._optimism_errors

    @property
    def consistency_errors(self) -> List[ConsistencyError]:
        return self._consistency_errors

    @property
    def is_optimistic(self):
        return len(self._optimism_errors) == 0

    @property
    def is_consistent(self):
        return len(self._consistency_errors) == 0

    def stringify_minimized(self):
        result = 'Heuristic is consistent.\n' if self.is_consistent else \
            'Heuristic is not consistent. Consistency errors:\n'
        for consistency_error in self.consistency_errors[:2]:
            result += str(consistency_error) + '\n'
        result += str(len(self.consistency_errors[2:])) + ' more' + '...\n'

        result += '\nHeuristic is optimistic.' if self.is_optimistic and self.is_consistent else \
            '\nHeuristic is not optimistic.\n' if self.is_optimistic and not self.is_consistent else \
                '\nHeuristic is not optimistic. Optimism errors:\n'
        for optimism_error in set(self.optimism_errors[:2]):
            result += str(optimism_error) + '\n'

        return result + str(len(self.optimism_errors[2:])) + ' more' + '...\n'

    def __str__(self):
        result = 'Heuristic is consistent.\n' if self.is_consistent else \
            'Heuristic is not consistent. Consistency errors:\n'
        for consistency_error in self.consistency_errors:
            result += str(consistency_error) + '\n'

        result += '\nHeuristic is optimistic.' if self.is_optimistic and self.is_consistent else \
            '\nHeuristic is not optimistic.\n' if self.is_optimistic and not self.is_consistent else \
            '\nHeuristic is not optimistic. Optimism errors:\n'
        for optimism_error in set(self.optimism_errors):
            result += str(optimism_error) + '\n'

        return result


class HeuristicQualityChecker:
    _shortest_path_search: AbstractSearchAlgorithm
    _data_manager: InformedDataManager

    def __init__(self, data_manager: InformedDataManager):
        self.data_manager = data_manager

    @property
    def data_manager(self):
        return self._data_manager

    @data_manager.setter
    def data_manager(self, data_manager: InformedDataManager) -> None:
        self._data_manager = deepcopy(data_manager)
        self._shortest_path_search = SearchAlgorithmFactory.create_blind_algorithm(
            BlindSearchAlgorithmKey.ShortestPathSearch, self.data_manager, SearchAlgorithmAttributes())

    def _check_consistency(self, result: HeuristicQualityCheckResult):
        for state_name in self.data_manager.state_space:
            for connected_state_name in self.data_manager.state_space[state_name]:
                transition_cost = self.data_manager.state_space[state_name][connected_state_name]
                state_heuristic = self.data_manager.heuristic[state_name]
                connected_state_heuristic = self.data_manager.heuristic[connected_state_name]

                if state_heuristic > connected_state_heuristic + transition_cost:
                    result.consistency_errors.append(
                        ConsistencyError(State(state_name, state_heuristic),
                                         State(connected_state_name, connected_state_heuristic),
                                         transition_cost))

    def _check_optimism(self, result: HeuristicQualityCheckResult):
        end_state_name = next((name for name in self.data_manager.goal.end_state_names if
                               self.data_manager.heuristic[name] == min(self.data_manager.heuristic.values())), None)
        if end_state_name is not None:
            self.data_manager.goal = Goal(end_state_name, [])
            treelike = self._shortest_path_search.search()
            for node in treelike:
                if node.state.heuristic > node.cost:
                    result.optimism_errors.append(OptimismError(node))

    def check_quality(self) -> HeuristicQualityCheckResult:
        result = HeuristicQualityCheckResult()
        self._check_consistency(result)
        self._check_optimism(result)
        return result
