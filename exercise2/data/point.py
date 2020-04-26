from typing import *
from dataclasses import dataclass

from data_structures.clause import Clause
from data_structures.tasks import Task


@dataclass(frozen=True)
class ParsedDataPoint:
    clauses: Iterable[Clause]
    tasks: Optional[Iterable[Task]]


class DataPoint:
    comment_character = '#'

    def parse(self) -> ParsedDataPoint:
        """
        >>> parsed = DataPoint('./files/cooking_examples/chicken_alfredo.txt', \
                               './files/cooking_examples/chicken_alfredo_input.txt').parse()
        >>> list(parsed.clauses)[0]
        Pasta
        >>> list(parsed.tasks)[2]
        Butter -
        """
        clauses = DataPoint._parse_clauses(self._clause_lines)

        try:
            return ParsedDataPoint(clauses, DataPoint._parse_tasks(self._task_lines))
        except AttributeError:
            return ParsedDataPoint(clauses, None)

    def __init__(self, clause_file_path: str, task_file_path: Optional[str] = None):
        self._clause_lines = self._get_file_lines(clause_file_path)
        if task_file_path is not None:
            self._task_lines = self._get_file_lines(task_file_path)

    @staticmethod
    def _get_file_lines(file_path: str) -> List[str]:
        with open(file_path) as file:
            return file.readlines()

    @staticmethod
    def _filter_comments(lines: Iterable[str]) -> Iterable[str]:
        return filter(lambda line: not line.startswith(DataPoint.comment_character), lines)

    @staticmethod
    def _parse_clauses(lines: Iterable[str]) -> Iterable[Clause]:
        return list(map(lambda line: Clause.parse(line[:-1]), DataPoint._filter_comments(lines)))

    @staticmethod
    def _parse_tasks(lines: Iterable[str]) -> Iterable[Task]:
        return list(map(lambda line: Task.parse(line[:-1]), DataPoint._filter_comments(lines)))
