from enum import Enum
from typing import Iterable, List
from os import path


class InputKey(Enum):
    three = '3x3'

    ai = 'ai'
    ai_pass = 'ai_pass'
    ai_fail = 'ai_fail'

    istra = 'istra'
    istra_normal = 'istra_normal'
    istra_pessimistic = 'istra_pessimistic'


class InputFileManager:
    @classmethod
    def get_state_space_lines(cls, key: InputKey) -> Iterable[str]:
        filtered = list(cls._filter_comments(cls._get_graph_file_lines(key)))
        return filtered[2:]

    @classmethod
    def get_search_goal_lines(cls, key: InputKey) -> Iterable[str]:
        filtered = list(cls._filter_comments(cls._get_graph_file_lines(key)))
        return filtered[:2]

    @classmethod
    def get_heuristics_lines(cls, key: InputKey) -> Iterable[str]:
        filtered = cls._filter_comments(cls._get_heuristics_file_lines(key))
        return filtered

    _relative_directory_path = path.join('.', 'input', 'files')
    _heuristics_key = 'heuristics'
    _normal_heuristic_key = 'normal'
    _graph_key = 'graph'
    _file_names = \
        {
            InputKey.three.value:
                {
                    _heuristics_key:
                        {
                            _normal_heuristic_key: '3x3_misplaced_heuristic.txt'
                        },
                    _graph_key: '3x3_puzzle.txt'
                },
            InputKey.ai.value:
                {
                    _heuristics_key:
                        {
                            _normal_heuristic_key: 'ai_pass.txt',
                            'fail': 'ai_fail.txt',
                            'pass': 'ai_pass.txt'
                        },
                    _graph_key: 'ai.txt'
                },
            InputKey.istra.value:
                {
                    _heuristics_key:
                        {
                            _normal_heuristic_key: 'istra_heuristic.txt',
                            'pessimistic': 'istra_pessimistic_heuristic.txt'
                        },
                    _graph_key: 'istra.txt'
                }
        }

    @classmethod
    def _get_graph_file_lines(cls, key: InputKey) -> Iterable[str]:
        state_space_name = key.value.split('_')[0]
        file_name = cls._file_names[state_space_name][cls._graph_key]
        lines = cls._get_data_file_lines(file_name)
        return lines

    @classmethod
    def _get_heuristics_file_lines(cls, key: InputKey) -> Iterable[str]:
        split_key: List[str] = key.value.split('_')
        heuristic_name = cls._normal_heuristic_key if len(split_key) == 1 else split_key[1]
        data_file_names = cls._file_names[split_key[0]]
        lines = cls._get_data_file_lines(data_file_names[cls._heuristics_key][heuristic_name])
        return lines

    @classmethod
    def _filter_comments(cls, lines: Iterable[str]) -> Iterable[str]:
        return filter(lambda line: not cls._is_comment(line), lines)

    @staticmethod
    def _is_comment(line):
        return line[0] == '#'

    @classmethod
    def _get_data_file_lines(cls, name: str) -> List[str]:
        with open(path.join(cls._relative_directory_path, name), encoding='utf') as data_file:
            return data_file.readlines()
