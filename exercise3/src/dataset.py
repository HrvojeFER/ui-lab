from __future__ import annotations

import os
from enum import *
from typing import *

import extensions


class Dataset(FrozenSet["Dataset.Row"]):
    """
    Represents a dataset used for machine learning purposes.
    It has interoperability with csv files and acts a bit as an SQL table with its columns and rows.
    """

    class Column(FrozenSet[Hashable], Hashable):
        """
        Represents a :class:`feature_column<dataset.Dataset.Column>` of a :class:`dataset<dataset.Dataset>` and
        stores the supported values and/or types of that feature_column.
        If the value frequency is discrete it will store the provided supported values for
        the :class:`feature_column<dataset.Dataset.Column>`, but if the value frequency is continuous it will store the
        types of the provided supported values. If the value frequency is set to automatic or left default
        it will store the type of the provided supported values if there is only one type of supported values and
        if that type is not a string.
        """

        class ValueFrequency(Enum):
            Auto = 'Auto'
            Discrete = 'Discrete'
            Continuous = 'Continuous'

        def __new__(cls,
                    name: str,
                    supported: Iterable[Hashable],
                    value_frequency: ValueFrequency = ValueFrequency.Auto):
            if value_frequency != Dataset.Column.ValueFrequency.Discrete:
                types = set()
                for value in supported:
                    value_type = type(value)
                    types.add(value_type if not isinstance(value, type) else value)

                if value_frequency == Dataset.Column.ValueFrequency.Continuous:
                    supported = types
                elif value_frequency == Dataset.Column.ValueFrequency.Auto:
                    first_type = next(iter(types))
                    if len(types) == 1 and first_type is not str:
                        supported = {first_type}

            # noinspection PyArgumentList
            return super(Dataset.Column, cls).__new__(cls, supported)

        # noinspection PyUnusedLocal
        def __init__(self,
                     name: str,
                     supported: Iterable[Hashable],
                     value_frequency: ValueFrequency = ValueFrequency.Auto):
            super().__init__()
            self._name: str = name

        @property
        def name(self) -> str:
            return self._name

        @property
        def supported(self) -> FrozenSet[Hashable]:
            return frozenset(self)

        def supports_value_or_type(self, value: Hashable) -> bool:
            return value in self or type(value) in self

        @property
        def values(self) -> FrozenSet[Hashable]:
            return self.supported

        class UnsupportedValueError(ValueError):
            def __init__(self, column: Dataset.Column, value: Any):
                super().__init__(f"Unsupported value ({repr(value)}) for feature_column ({repr(column)}).")

        def __eq__(self, other: Dataset.Column) -> bool:
            return self._name == other._name

        def __hash__(self) -> int:
            return hash(self._name)

        _string_value_separator: str = ':'

        def __str__(self) -> str:
            return f'{self._name}'

        def __repr__(self) -> str:
            return f'{self._name}{self._string_value_separator} {set(self)}'

        @classmethod
        def parse(cls, string_value: str, value_type: ValueFrequency = ValueFrequency.Auto) -> Dataset.Column:
            """
            >>> repr(Dataset.Column.parse("A: (0, '1')"))
            "A: {0, '1'}"

            >>> repr(Dataset.Column.parse("A: (0, 1)"))
            "A: {<class 'int'>}"


            :param string_value: name of the feature_column and
            the string representation of a Python collection separated by a colon (': ')
            :param value_type: use discrete, continuous or decide automatically which value type to use

            :return: parsed feature_column with the inferred possible value types.
            """

            split = string_value.strip().split(cls._string_value_separator)
            name = split[0].strip()
            feature_values = extensions.safe_literal_eval(split[1].strip())
            return Dataset.Column(name, feature_values, value_type)

    class Row(Mapping["Dataset.Column", Hashable], Hashable):
        """
        Represents a row of a :class:`dataset<dataset.Dataset>` and stores its values as a mapping of
        :class:`feature_column<dataset.Dataset.Column>` to value.
        """

        class ColumnValue(Hashable):
            """
            Represents the value of a :class:`feature_column<dataset.Dataset.Column>`
            in a :class:`row<dataset.Dataset.Row>` of a :class:`dataset<dataset.Dataset>` and
            keeps a reference of that :class:`feature_column<dataset.Dataset.Column>`.
            """

            def __init__(self, column: Dataset.Column, value: Hashable):
                self._column: Dataset.Column = column
                self._value: Hashable = value

                if not self._column.supports_value_or_type(value):
                    raise Dataset.Column.UnsupportedValueError(self._column, value)

            @property
            def column(self) -> Dataset.Column:
                return self._column

            @property
            def value(self) -> Hashable:
                return self._value

            def __eq__(self, other: Dataset.Row.ColumnValue[Dataset.Column]) -> bool:
                return self._value == other._value

            def __hash__(self) -> int:
                return hash(self._column)

            _string_value_separator = '->'

            def __str__(self) -> str:
                return f"{self._value}"

            def __repr__(self) -> str:
                return f"{repr(self._column)} {self._string_value_separator} {repr(self._value)}"

            @classmethod
            def parse(cls, string_value: str) -> Dataset.Row.ColumnValue[Dataset.Column]:
                """
                >>> repr(Dataset.Row.ColumnValue.parse("A: (0, '1') -> '1'"))
                "A: {0, '1'} -> '1'"

                :param string_value: feature_column string representation and
                the string of the value separated by a right arrow (' -> ')
                :return: parsed feature_column value with inferred possible feature_column value types and
                inferred value type

                :raises UnsupportedValueError: when the value isn't one of the supported feature_column values or
                doesn't match one of the supported types.
                """

                split = string_value.strip().split(cls._string_value_separator)
                column = Dataset.Column.parse(split[0].strip())
                value = extensions.safe_literal_eval(split[1].strip())
                return Dataset.Row.ColumnValue(column, value)

        def __init__(self,
                     values: Union[Iterable[Dataset.Row.ColumnValue], Mapping[Dataset.Column, Hashable]]):
            if isinstance(values, Mapping):
                self._columns_to_values: Dict[Dataset.Column, Hashable] = dict(values)
            else:
                self._columns_to_values: Dict[Dataset.Column, Hashable] = \
                    dict((column_value.column, column_value.value) for column_value in values)

            for column, value in self._columns_to_values.items():
                if not column.supports_value_or_type(value):
                    raise Dataset.Column.UnsupportedValueError(column, value)

        @property
        def column_values(self) -> FrozenSet[Dataset.Row.ColumnValue]:
            return frozenset(Dataset.Row.ColumnValue(column, value)
                             for column, value in self._columns_to_values.items())

        @property
        def columns(self) -> FrozenSet[Dataset.Column]:
            return frozenset(column for column in self._columns_to_values.keys())

        def get_value_for(self, column: Dataset.Column) -> Hashable:
            return self._columns_to_values[column]

        def __getitem__(self, column: Dataset.Column) -> Hashable:
            return self.get_value_for(column)

        def __len__(self) -> int:
            return len(self._columns_to_values)

        def __iter__(self) -> Iterator[Dataset.Row.ColumnValue]:
            return iter(self.column_values)

        def select(self, *columns: Dataset.Column) -> Dataset.Row:
            to_select = set(columns).intersection(self._columns_to_values.keys())
            return Dataset.Row(Dataset.Row.ColumnValue(column, self[column]) for column in to_select)

        def __eq__(self, other: Dataset.Row) -> bool:
            return self._columns_to_values == other._columns_to_values

        def __hash__(self) -> int:
            return sum(hash(value) for value in self._columns_to_values.values())

        _string_value_separator = ','

        def __str__(self) -> str:
            return self._string_value_separator.join(str(value) for value in self._columns_to_values.values())

        def __repr__(self) -> str:
            return f'{self._string_value_separator} '.join(
                f'{str(column)}: {repr(value)}' for column, value in self._columns_to_values.items())

        class InvalidRowValuesError(ValueError):
            _column_separator = '\t\n'

            def __init__(self, columns: List[Dataset.Column], values: List[Any]):
                super().__init__(f"Unsupported values ({repr(values)}) for columns:"
                                 f"\t{self._column_separator.join(repr(column) for column in columns)}")

        @classmethod
        def parse(cls, columns: List[Dataset.Column], string_value: str) -> Dataset.Row:
            """
            >>> repr(Dataset.Row.parse([
            ...     Dataset.Column.parse("A: (0, '1')"),
            ...     Dataset.Column.parse("B: (3, 'six')")],
            ...     "0, six"))
            "A: 0, B: 'six'"


            :param columns: columns that the values of this row will represent.
            :param string_value: values of the row separated with a comma and space (', ')

            :return: parsed row with inferred value types

            :raises InvalidRowValuesError: when the values aren't one of the supported feature_column values or
            when they don't match the supported feature_column types.
            """
            values = string_value.strip().split(cls._string_value_separator)
            if len(columns) != len(values):
                raise Dataset.Row.InvalidRowValuesError(columns, values)

            try:
                return Dataset.Row(dict((column, extensions.safe_literal_eval(value.strip()))
                                        for column, value in zip(columns, values)))
            except Dataset.Column.UnsupportedValueError:
                raise Dataset.Row.InvalidRowValuesError(columns, values)

    class NonMatchingRowsAndColumnsError(ValueError):
        _column_row_separator = '\t\n'

        def __init__(self,
                     columns: Iterable[Dataset.Column],
                     rows: Iterable[Dataset.Row[Dataset.Column, Dataset.Row.ColumnValue[Dataset.Column]]]):
            super().__init__(f"Rows:"
                             f"\t{self._column_row_separator.join(str(row) for row in rows)}"
                             f"don't match the columns:"
                             f"\t{self._column_row_separator.join(str(column) for column in columns)}.")

    def __new__(cls, name: str, columns: Iterable[Dataset.Column], rows: Iterable[Dataset.Column]):
        # noinspection PyArgumentList
        return super(Dataset, cls).__new__(cls, rows)

    # noinspection PyUnusedLocal
    def __init__(self, name: str, columns: Iterable[Dataset.Column], rows: Iterable[Dataset.Row]):
        super().__init__()
        self._name: str = name

        # Using dictionary keys for keeping things in order.
        # I might add an ordered set extension for this.
        columns = dict.fromkeys(columns)
        self._columns: Dict[Dataset.Column, None] = columns

        column_set = self.columns
        for row in self:
            if row.columns != column_set:
                raise Dataset.NonMatchingRowsAndColumnsError(self._columns, self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def columns(self) -> FrozenSet[Dataset.Column]:
        return frozenset(self._columns.keys())

    @property
    def rows(self) -> FrozenSet[Dataset.Row]:
        return frozenset(self)

    @property
    def feature_columns_tuple(self) -> Tuple[Dataset.Column, ...]:
        """
        >>> column_a = Dataset.Column.parse("A: {'a'}")
        >>> column_b = Dataset.Column.parse("B: {'b'}")
        >>> row = Dataset.Row({column_a: 'a', column_b: 'b'})
        >>> dataset = Dataset("My Dataset", [column_a, column_b], {row})
        >>> repr(dataset.feature_columns_tuple)
        "(A: {'a'},)"

        :return: all but the last feature_column in the dataset, which represent features of the dataset
        """

        return tuple(self._columns.keys())[:-1]

    @property
    def feature_columns(self) -> FrozenSet[Dataset.Column]:
        return frozenset(self.feature_columns_tuple)

    @property
    def result_column(self) -> Dataset.Column:
        """
        >>> column_a = Dataset.Column.parse("A: {'a'}")
        >>> column_b = Dataset.Column.parse("B: {'b'}")
        >>> row = Dataset.Row({column_a: 'a', column_b: 'b'})
        >>> dataset = Dataset("My Dataset", [column_a, column_b], {row})
        >>> repr(dataset.result_column)
        "B: {'b'}"

        :return: the last feature_column of the dataset representing results of rows in the dataset
        """

        return list(self._columns.keys())[-1]

    def select(self, *columns: Dataset.Column) -> Dataset:
        """
        >>> column_a = Dataset.Column.parse("A: {'a'}")
        >>> column_b = Dataset.Column.parse("B: {'b'}")
        >>> row = Dataset.Row({column_a: 'a', column_b: 'b'})
        >>> dataset = Dataset("My Dataset", {column_a, column_b}, {row})
        >>> repr(dataset.select(column_a))
        'A\\na'

        :param columns: columns to select
        :return: a dataset with the selected columns
        """
        if len(columns) > 0:
            to_select = set(columns).intersection(self._columns.keys())
            return Dataset(self._name, to_select, (row.select(*to_select) for row in self))
        else:
            return Dataset(self._name, self._columns.keys(), self)

    def where(self, column_value: Dataset.Row.ColumnValue) -> Dataset:
        """
        >>> column_a = Dataset.Column.parse("A: {'a', 'c'}")
        >>> column_b = Dataset.Column.parse("B: {'b'}")
        >>> row_1 = Dataset.Row({column_a: 'a', column_b: 'b'})
        >>> row_2 = Dataset.Row({column_a: 'c', column_b: 'b'})
        >>> dataset = Dataset("My Dataset", [column_a, column_b], {row_1, row_2})
        >>> repr(dataset.where(Dataset.Row.ColumnValue(column_a, 'c')))
        'A,B\\nc,b'

        :param column_value:
        :return:
        """
        return Dataset(
            self._name,
            self._columns.keys(),
            (row for row in self if row[column_value.column] == column_value.value))

    _column_separator = ','
    _row_separator = '\n'

    def is_empty(self) -> bool:
        return len(self) == 0

    def is_nil(self) -> bool:
        return len(self._columns.keys()) == 0

    def __str__(self) -> str:
        return f"{self._column_separator.join(str(column) for column in self._columns.keys())}" \
               f"{self._row_separator}" \
               f"{self._row_separator.join(str(row) for row in self)}"

    def __repr__(self) -> str:
        return f"{self._column_separator.join(str(column) for column in self._columns.keys())}" \
               f"{self._row_separator}" \
               f"{self._row_separator.join(str(row) for row in self)}"

    @classmethod
    def parse(cls,
              name: str,
              string_value: str,
              *column_value_frequencies: Dataset.Column.ValueFrequency) -> Dataset:
        """
        >>> repr(Dataset.parse("My Dataset", "A, B\\na, b"))
        'A,B\\na,b'


        :param name: name of the dataset
        :param string_value: rows with values separated by a comma (', ') with
        the first row representing the feature_column names.

        :return: parsed dataset

        :raises InvalidRowValuesError: if a row is not the same length as the feature_column count
        """

        string_value = string_value.strip()
        lines_for_columns: List[List[str]] = []
        for line in string_value.split(cls._row_separator):
            lines_for_columns.append([value.strip() for value in line.strip().split(cls._column_separator)])

        columns = []
        if len(column_value_frequencies) == 0:
            column_value_frequencies = tuple(Dataset.Column.ValueFrequency.Auto for _ in lines_for_columns[0])
        elif len(column_value_frequencies) != len(lines_for_columns[0]):
            column_value_frequencies = tuple(column_value_frequencies[0] for _ in lines_for_columns[0])
        for index, column_name in enumerate(lines_for_columns[0]):
            columns.append(Dataset.Column(
                column_name,
                set(extensions.safe_literal_eval(row[index]) for row in lines_for_columns[1:]),
                column_value_frequencies[index]))

        lines_for_rows = [line.strip() for line in string_value.split(cls._row_separator)[1:]]
        rows = set()
        for line in lines_for_rows:
            rows.add(Dataset.Row.parse(columns, line))

        return Dataset(name, columns, rows)

    test_file_path = os.path.join("..", "data", "datasets", "logic_small.csv")

    @classmethod
    def from_csv_file(cls,
                      path_to_csv_file: str,
                      *column_value_frequencies: Dataset.Column.ValueFrequency) -> Dataset:
        """
        >>> #Dummy tests that should always work, but they are there for debugging purposes
        >>> os.path.exists(Dataset.test_file_path)
        True

        >>> len(repr(Dataset.from_csv_file(Dataset.test_file_path))) > 0
        True


        :param path_to_csv_file: path to csv file
        :return: parsed dataset from file, as documented in the parse method
        """

        with open(path_to_csv_file) as csv_file:
            return cls.parse(os.path.basename(path_to_csv_file), csv_file.read(), *column_value_frequencies)
