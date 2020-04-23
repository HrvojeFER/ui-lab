from typing import *
from data_structures.clause import NormalClause
from data_structures.commands import NormalCommand


class DataFileParser:
    comment_character = '#'

    @classmethod
    def filter_comments(cls, lines: Iterable[str]) -> Iterable[str]:
        return filter(lambda line: line.startswith(cls.comment_character), lines)


class ClauseParser(DataFileParser):
    @classmethod
    def parse(cls, lines: Iterable[str]) -> List[NormalClause]:
        return list(map(lambda line: NormalClause(line), cls.filter_comments(lines)))


class CommandParser(DataFileParser):
    @classmethod
    def parse(cls, lines: Iterable[str]) -> List[NormalCommand]:
        return list(map(lambda line: NormalCommand(line), cls.filter_comments(lines)))
