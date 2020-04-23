from typing import *


class DataPoint:
    def __init__(self, clause_file_path: str, command_file_path: Optional[str] = None):
        self.clause_lines = self._get_file_lines(clause_file_path)
        if command_file_path is not None:
            self.command_lines = self._get_file_lines(command_file_path)

    @staticmethod
    def _get_file_lines(file_path: str) -> List[str]:
        with open(file_path) as file:
            return file.readlines()
