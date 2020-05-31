import ast

from typing import *

ArgType = TypeVar('ArgType', contravariant=True)


def arg_max(predicate: Callable[[ArgType], Any], arguments: Iterable[ArgType]):
    # noinspection PyUnresolvedReferences
    """
    >>> arg_max(lambda x: x * 2, [1, 2, 3])
    3

    :param predicate: Function that takes the ArgType and returns a value that supports the '>' operator.
    :param arguments: Arguments to test the predicate with.
    :return: The argument that has the maximum value when applying the predicate to it.
    """
    arguments_iterator = iter(arguments)
    result: str = next(arguments_iterator)
    predicate_result: Any = predicate(result)

    for test in arguments_iterator:
        current_predicate_result = predicate(test)
        if predicate_result is None or current_predicate_result > predicate_result:
            result = test
            predicate_result = current_predicate_result

    return result


def safe_literal_eval(literal: str) -> Any:
    """
    >>> safe_literal_eval('a')
    'a'

    :param literal: The string to evaluate.
    :return: If the string can be evaluated as a literal, returns a literal evaluated from it,
    otherwise returns the string.
    """
    try:
        return ast.literal_eval(literal)
    except ValueError:
        return literal
