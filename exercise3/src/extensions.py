import ast

from typing import *

ArgType = TypeVar('ArgType', contravariant=True)


def arg_max(iterable: Iterable[ArgType], *iterables: Iterable[ArgType], key: Callable[[ArgType], Any]) -> \
        Optional[ArgType]:
    # noinspection PyUnresolvedReferences
    """
    >>> arg_max([1, 2, 3], key=lambda x: x * 2)
    3

    >>> arg_max([], key=lambda x: x * 2)

    >>> arg_max([1], [2, 3], [], key=lambda x: x * 2)
    3

    :param key: function that takes the ArgType and returns a value that supports the '>' operator.
    :param iterable: arguments to test the key with
    :param iterables: iterables of arguments to test the key with
    :return: the argument that has the maximum value when applying the key to it or None if the iterables are empty
    """
    iterables = (iterable, *iterables)

    result: Optional[Any] = None
    predicate_result: Optional[Any] = None

    for iterable in iterables:
        for test in iterable:
            current_predicate_result = key(test)
            if predicate_result is None or current_predicate_result > predicate_result:
                result = test
                predicate_result = current_predicate_result

    return result


def safe_literal_eval(literal: str) -> Any:
    """
    >>> safe_literal_eval('a')
    'a'

    >>> safe_literal_eval("[0, 2, 1]")
    [0, 2, 1]

    :param literal: The string to evaluate.
    :return: If the string can be evaluated as a literal, returns a literal evaluated from it,
    otherwise returns the string.
    """
    try:
        return ast.literal_eval(literal)
    except ValueError:
        return literal
    except SyntaxError:
        return literal
