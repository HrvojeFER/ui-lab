import ast
from typing import *


ValueType = TypeVar('ValueType')


def flatten(*iterables_or_values: Any,
            nesting_limit: int = 100,
            type_limits: Optional[Iterable[type]] = None) -> Generator[Any, None, None]:
    """
    >>> list(flatten())
    []

    >>> list(flatten([]))
    []

    >>> list(flatten([1, 2, 3]))
    [1, 2, 3]

    >>> list(flatten(1, 2, 3))
    [1, 2, 3]

    >>> list(flatten([1, 2, [3]]))
    [1, 2, 3]

    >>> list(flatten([1, [2, 3], 4], [5, 6, [7, 8, [9]]], 10, 11, [12, 13]))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    >>> list(flatten([1, 2, {3}], type_limits={set}))
    [1, 2, {3}]

    >>> list(flatten([1, 2, 3], type_limits={list}))
    [[1, 2, 3]]

    :param iterables_or_values: iterables or values or values to flatten
    :param nesting_limit: to avoid recursion
    :param type_limits: types not to flatten

    :return: flattened iterables or values or values
    """

    if nesting_limit == 0:
        return

    if not isinstance(type_limits, set):
        if type_limits is None:
            type_limits: Set[type] = set()
        else:
            type_limits: Set[type] = set(type_limits)

    def is_qualified_iterable(value: Any) -> bool:
        return isinstance(value, Iterable) and not \
                isinstance(value, str) and not \
                isinstance(value, bytes) and not \
                (type(value) in type_limits)

    if len(iterables_or_values) == 1:
        if is_qualified_iterable(iterables_or_values[0]):
            for iterable_or_value in iterables_or_values[0]:
                yield from flatten(iterable_or_value, nesting_limit=nesting_limit, type_limits=type_limits)
        else:
            yield iterables_or_values[0]
    elif len(iterables_or_values) > 1:
        for iterable_or_value in iterables_or_values:
            yield from flatten(iterable_or_value, nesting_limit=nesting_limit - 1, type_limits=type_limits)


def get(*iterables_or_values: Any,
        at: int,
        type_limits: Optional[Set[type]] = None) -> Any:
    """
    >>> get([1, 2, 3], at=0)
    1

    >>> get([1, 2, 3], 10, [1, 2, 3, 4], at=3)
    10

    :param iterables_or_values: values
    :param at: index to get the value at
    :param type_limits: which types not to flatten

    :return: last value in the iterables_or_values
    """

    flattened = flatten(iterables_or_values, type_limits=type_limits)

    if at >= 0:
        for index, iterable_value in enumerate(flattened):
            if index == at:
                return iterable_value

    else:
        return tuple(flatten(iterables_or_values, type_limits=type_limits))[at]

    raise IndexError(f"Nothing at ({at})")


def max_argument(predicate: Callable[[ValueType], Any],
                 *iterables_or_values: Any,
                 type_limits: Optional[Set[type]] = None) -> Optional[ValueType]:
    # noinspection PyUnresolvedReferences
    """
    >>> max_argument(lambda x: x * 2, [1, 2.2, [3]])
    3

    >>> max_argument(lambda x: x ** 2, 4, 6, 5)
    6

    :param predicate: function that takes the ArgType and returns a value_or_type that supports the '>' operator.
    :param iterables_or_values: iterables_or_values or arguments to test the key with
    :param type_limits: which types not to flatten
    :return: the argument that has the maximum value_or_type when applying the key to it or None if
    the iterables_or_values are empty
    """

    result: Optional[ValueType] = None
    predicate_result: Optional[Any] = None

    for value in flatten(iterables_or_values, type_limits=type_limits):
        current_predicate_result = predicate(value)
        if predicate_result is None or current_predicate_result > predicate_result:
            result = value
            predicate_result = current_predicate_result

    return result


def get_first(*iterables_or_values: Any, type_limits: Optional[Set[type]] = None) -> Optional[Any]:
    """
    >>> get_first([1, 2, 3])
    1

    >>> get_first([1, 2, 3], [2, 3, 4])
    1

    >>> get_first([])

    >>> get_first()


    :param iterables_or_values: values
    :param type_limits: which types not to flatten

    :return: first value in iterables_or_values
    """

    for value in flatten(iterables_or_values, type_limits=type_limits):
        return value


def get_last(*iterables_or_values: Any, type_limits: Optional[Set[type]] = None) -> Optional[Any]:
    """
    >>> get_last([1, 2, 3])
    3

    >>> get_last([1, 2, 3], [1, 2, 3, 4])
    4

    >>> get_last([])

    >>> get_last()


    :param iterables_or_values: values
    :param type_limits: which types not to flatten

    :return: last value in the iterables_or_values
    """

    value: Optional[ValueType] = None

    for iterable_value in flatten(iterables_or_values, type_limits=type_limits):
        value = iterable_value

    return value


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
