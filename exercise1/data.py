from typing import List, Dict, Mapping, Iterator, Iterable, Optional
from enum import Enum
from dataclasses import dataclass


class DataKeys(Enum):
    three = '3x3'

    ai = 'ai'
    ai_pass = 'ai_pass'
    ai_fail = 'ai_fail'

    istra = 'istra'
    istra_normal = 'istra_normal'
    istra_pessimistic = 'istra_pessimistic'


class DataManager:
    _directory_name = 'data'
    _heuristics_key = 'heuristics'
    _normal_heuristic_key = 'normal'
    _graph_key = 'graph'
    _file_names = \
        {
            DataKeys.three.value:
                {
                    _heuristics_key:
                        {
                            _normal_heuristic_key: '3x3_misplaced_heuristic.txt'
                        },
                    _graph_key: '3x3_puzzle.txt'
                },
            DataKeys.ai.value:
                {
                    _heuristics_key:
                        {
                            _normal_heuristic_key: 'ai_pass.txt',
                            'fail': 'ai_fail.txt',
                            'pass': 'ai_pass.txt'
                        },
                    _graph_key: 'ai.txt'
                },
            DataKeys.istra.value:
                {
                    _heuristics_key:
                        {
                            _normal_heuristic_key: 'istra_heuristic.txt',
                            'pessimistic': 'istra_pessimistic_heuristic.txt'
                        },
                    _graph_key: 'istra.txt'
                }
        }

    @staticmethod
    def get_heuristics_lines(key: DataKeys) -> Iterable[str]:
        filtered = DataManager._filter_comments(DataManager._get_heuristics_file_lines(key))
        return filtered

    @staticmethod
    def _get_heuristics_file_lines(key: DataKeys) -> Iterable[str]:
        split_key: List[str] = key.value.split('_')
        heuristic_name = DataManager._normal_heuristic_key if len(split_key) == 1 else split_key[1]
        data_file_names = DataManager._file_names[split_key[0]]
        lines = DataManager._get_data_file_lines(data_file_names[DataManager._heuristics_key][heuristic_name])
        return lines

    @staticmethod
    def get_state_space_lines(key: DataKeys) -> Iterable[str]:
        filtered = list(DataManager._filter_comments(DataManager._get_graph_file_lines(key)))
        return filtered[2:]

    @staticmethod
    def get_search_goal_lines(key: DataKeys) -> Iterable[str]:
        filtered = list(DataManager._filter_comments(DataManager._get_graph_file_lines(key)))
        return filtered[:2]

    @staticmethod
    def _get_graph_file_lines(key: DataKeys) -> Iterable[str]:
        state_space_name = key.value.split('_')[0]
        file_name = DataManager._file_names[state_space_name][DataManager._graph_key]
        lines = DataManager._get_data_file_lines(file_name)
        return lines

    @staticmethod
    def _filter_comments(lines: Iterable[str]) -> Iterable[str]:
        return filter(lambda line: not DataManager._is_comment(line), lines)

    @staticmethod
    def _is_comment(line):
        return line[0] == '#'

    @staticmethod
    def _get_data_file_lines(name: str) -> List[str]:
        with open('./' + DataManager._directory_name + '/' + name, encoding='utf') as data_file:
            return data_file.readlines()


@dataclass
class StateSpace(Mapping[str, Mapping[str, int]]):
    _graph: Dict[str, Dict[str, int]]

    @staticmethod
    def parse(lines: Iterable[str]):
        graph: Dict[str, Dict[str, int]] = {}

        for line in lines:
            split_line = line.split()
            state_name = split_line[0][:-1]

            for transition_str in split_line[1:]:
                split_transition_str = transition_str.split(',')
                try:
                    graph[state_name][split_transition_str[0]] = int(split_transition_str[1])
                except KeyError:
                    graph[state_name] = {split_transition_str[0]: int(split_transition_str[1])}

        return StateSpace(graph)

    def __getitem__(self, state_name: str) -> Dict[str, int]:
        return self._graph[state_name]

    def __len__(self) -> int:
        return len(self._graph)

    def __iter__(self) -> Iterator[str]:
        return iter(self._graph)

    def __str__(self) -> str:
        result = ''

        for state_name in self:
            result += state_name + ': '
            for transition_state_name in self[state_name]:
                result += transition_state_name + ',' + str(self[state_name][transition_state_name]) + ' '
            result += '\n'

        return result


@dataclass
class Heuristic(Mapping[str, int]):
    _inner: Dict[str, int]

    @staticmethod
    def parse(lines: Iterable[str]):
        inner: Dict[str, int] = {}

        for line in lines:
            split_line: List[str] = line.split(': ')
            inner[split_line[0]] = int(split_line[1])

        return Heuristic(inner)

    def __getitem__(self, state_name: str) -> int:
        return self._inner[state_name]

    def __len__(self) -> int:
        return len(self._inner)

    def __iter__(self) -> Iterator[str]:
        return iter(self._inner)

    def __str__(self):
        result = ''

        for state_name in self:
            result += state_name + ': ' + str(self[state_name]) + '\n'

        return result


@dataclass
class SearchGoal:
    start_state_name: str
    end_state_names: List[str]

    @staticmethod
    def parse(lines: Iterable[str]):
        iterator = iter(lines)
        return SearchGoal(next(iterator)[:-1], next(iterator)[:-1].split())

    def __str__(self):
        return 'Start: ' + self.start_state_name + '\n' + \
                'End: ' + ' '.join(self.end_state_names) + '\n'


@dataclass
class State:
    _name: str
    heuristic: Optional[int]

    def get_name(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name + ('' if self.heuristic is None else (': ' + str(self.heuristic)))
