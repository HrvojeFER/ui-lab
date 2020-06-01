import math
from typing import *

from dataset import Dataset


def relative_frequency(dataset: Dataset, column: Dataset.Column, value: Any):
    """
    >>> dataset = Dataset.from_csv_file(Dataset.test_file_path)
    >>> relative_frequency(dataset, dataset.result_column, True)
    0.2


    :param dataset: dataset to perform the calculation on
    :param column: feature_column on which the value_or_type will be tested
    :param value: the value_or_type to test with

    :return: frequency of the rows that have the specified value_or_type of the specified feature_column
    """

    return len(dataset.where(Dataset.Row.ColumnValue(column, value))) / len(dataset)


def _entropy(dataset: Dataset, column: Dataset.Column, value: Any):
    """
    >>> dataset = Dataset.from_csv_file(Dataset.test_file_path)
    >>> _entropy(dataset, dataset.feature_column_tuple[3], True)
    0.5


    :param dataset: dataset to perform the calculation on
    :param column: feature_column on which the value_or_type will be tested
    :param value: the value_or_type to test with

    :return: entropy of the rows that have the specified value_or_type of the specified feature_column
    """

    probability = relative_frequency(dataset, column, value)
    return - probability * math.log2(probability) if probability != 0 else 0


def entropy(dataset: Dataset, column: Dataset.Column, *values: Any) -> float:
    """
    >>> dataset = Dataset.from_csv_file(Dataset.test_file_path, Dataset.Column.ValueFrequency.Discrete)
    >>> entropy(dataset, dataset.result_column, *dataset.result_column)
    0.7219280948873623


    :param dataset: dataset to perform the calculation on
    :param column: feature_column on which the value_or_type will be tested
    :param values: the row to test with

    :return: entropy of the rows that have the specified row of the specified feature_column
    """

    return sum(_entropy(dataset, column, value) for value in values)


def information_gain(dataset: Dataset, feature_column: Dataset.Column) -> float:
    """
    >>> dataset = Dataset.from_csv_file(Dataset.test_file_path, Dataset.Column.ValueFrequency.Discrete)
    >>> information_gain(dataset, dataset.result_column)
    0.7219280948873623


    :param dataset: dataset to perform the calculation on
    :param feature_column: feature_column feature_column for which the information gain will be calculated

    :return: information gain for the specified feature_column feature_column
    """

    result_column = dataset.result_column
    result_entropy = entropy(dataset, result_column, *result_column)

    def __feature_entropy_generator(
            dataset_: Dataset,
            feature_column_: Dataset.Column) -> Generator[float, None, None]:
        for value in feature_column_:
            reduced_dataset = dataset_.where(Dataset.Row.ColumnValue(feature_column, value))
            yield (entropy(reduced_dataset, dataset.result_column, *dataset.result_column) *
                   len(reduced_dataset) / len(dataset)) if len(reduced_dataset) != 0 else 0

    feature_entropy = sum(__feature_entropy_generator(dataset, feature_column))

    return result_entropy - feature_entropy
