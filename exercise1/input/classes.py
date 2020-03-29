from dataclasses import dataclass
from typing import Mapping, Dict, List, Iterable, Iterator


@dataclass(frozen=True)
class StateSpace(Mapping[str, Mapping[str, int]]):
    _graph: Dict[str, Dict[str, int]]

    @staticmethod
    def parse(lines: Iterable[str]):
        graph: Dict[str, Dict[str, int]] = {}

        for line in lines:
            split_line = line.split()
            state_name = split_line[0][:-1]

            if len(split_line[1:]) == 0:
                graph[state_name] = {}
                continue

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


@dataclass(frozen=True)
class Goal:
    start_state_name: str
    end_state_names: List[str]

    @staticmethod
    def parse(lines: Iterable[str]):
        iterator = iter(lines)
        return Goal(next(iterator)[:-1], next(iterator)[:-1].split())

    def __str__(self):
        return 'Start: ' + self.start_state_name + '\n' + \
                'End: ' + 'or'.join(self.end_state_names) + '\n'


@dataclass(frozen=True)
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
