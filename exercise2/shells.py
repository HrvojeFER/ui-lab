from enum import *
from abc import *
from typing import *

from data.point import DataPoint, ParsedDataPoint

from data_structures.tasks import Task, Inquiry
from data_structures.reports import Report

from algorithms.refutation_resolution import RefutationResolution
from algorithms.control_strategies import SupportSetStrategy
from algorithms.simplification_strategies import RedundantClauseRemoval, InsignificantClauseRemoval


class ShellMode(Enum):
    resolution_mode: str = 'resolution'
    cooking_test_mode: str = 'cooking_test'
    cooking_interactive_mode: str = 'cooking_interactive'


class Shell(ABC):
    _resolution: RefutationResolution
    _print_report: Callable[[Report], NoReturn]

    def __init__(self, verbose: bool = False):
        if verbose:
            self._print_report = Shell._print_report_verbose
        else:
            self._print_report = Shell._print_report_short

    @staticmethod
    def _print_report_short(report: Report) -> NoReturn:
        print(report.short_str())

    @staticmethod
    def _print_report_verbose(report: Report) -> NoReturn:
        print(report)

    @abstractmethod
    def run(self) -> NoReturn:
        ...


def shell_factory(mode: str,
                  clauses_file_path: str,
                  tasks_file_path: Optional[str] = None,
                  verbose: bool = False) -> 'Shell':
    try:
        shell_mode = ShellMode(mode)
    except ValueError:
        raise ValueError('%s is not a valid mode.' % mode)

    parsed = DataPoint(clauses_file_path, tasks_file_path).parse()
    if shell_mode is ShellMode.resolution_mode:
        return ResolutionShell(parsed, verbose)
    elif shell_mode is ShellMode.cooking_test_mode:
        return CookingTestShell(parsed, verbose)
    elif shell_mode is ShellMode.cooking_interactive_mode:
        return CookingInteractiveShell(parsed, verbose)


class ResolutionShell(Shell):
    def __init__(self, parsed: ParsedDataPoint, verbose: bool = False):
        super().__init__(verbose)
        clause_list = list(parsed.clauses)
        self._resolution = RefutationResolution(SupportSetStrategy(
            clause_list[:-1],
            clause_list[-1],
            simplification_strategies=(RedundantClauseRemoval(), InsignificantClauseRemoval())))

    def run(self) -> NoReturn:
        """
        >>> shell_factory( \
                ShellMode.resolution_mode, \
                './data/files/resolution_examples/chicken_broccoli_alfredo_big.txt', \
                verbose=True).run()
        """
        self._print_report(self._resolution.resolve())


class CookingShell(Shell, ABC):
    _tasks: List[Task]

    def __init__(self, parsed: ParsedDataPoint, verbose: bool = False):
        super().__init__(verbose)
        self._resolution = RefutationResolution(SupportSetStrategy(
            parsed.clauses,
            simplification_strategies=(RedundantClauseRemoval(), InsignificantClauseRemoval())))

        if parsed.tasks is not None:
            self._tasks = list(parsed.tasks)
        else:
            self._tasks = []


class CookingTestShell(CookingShell):
    def run(self) -> NoReturn:
        """
        >>> shell_factory( \
                ShellMode.cooking_test_mode, \
                './data/files/cooking_examples/chicken_alfredo_nomilk.txt', \
                './data/files/cooking_examples/chicken_alfredo_nomilk_input.txt', \
                verbose=True).run()
        """
        for task in self._tasks:
            report = self._resolution.do(task)
            if isinstance(task, Inquiry):
                self._print_report(report)


class CookingInteractiveShell(CookingTestShell):
    def run(self) -> NoReturn:
        super().run()
        print('Constructed with knowledge:\n%s' % '\n'.join(
            ('> %s' % str(premise) for premise in self._resolution.premises)))

        while True:
            print('Please enter your query.')

            user_input: str = input()
            if user_input == 'exit':
                break
            print()
            try:
                self._print_report(self._resolution.do(Task.parse(user_input)))
            except ValueError:
                print('Invalid query. Please try again.\n')
