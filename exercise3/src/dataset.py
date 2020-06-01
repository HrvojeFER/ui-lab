from __future__ import annotations

import os
from enum import *
from typing import *

from extensions import safe_literal_eval


class Dataset(Collection["Dataset.Row"]):
    """
    Represents a dataset used for machine learning purposes.
    It has interoperability with csv files and acts a bit as an SQL table with its columns and rows.
    """

    class Column(FrozenSet[Union[Hashable, None]], Hashable):
        """
        Represents a :class:`feature_column<dataset.Dataset.Column>` of a :class:`dataset<dataset.Dataset>` and
        stores the supported row and/or types of that feature_column.
        If the value frequency is discrete it will store the provided supported row for
        the :class:`feature_column<dataset.Dataset.Column>`, but if the value frequency is continuous it will store the
        types of the provided supported row. If the value frequency is set to automatic or left default
        it will store the type of the provided supported row if there is only one type of supported row and
        if that type is not a string.
        """

        class ValueFrequency(Enum):
            Auto = 'Auto'
            Discrete = 'Discrete'
            Continuous = 'Continuous'

        @staticmethod
        def aggregate_value_frequencies(value_frequencies: Iterable[Dataset.Column.ValueFrequency]) -> \
                Dataset.Column.ValueFrequency:
            result: Optional[Dataset.Column.ValueFrequency] = None

            for value_frequency in value_frequencies:
                if result is not None and \
                        value_frequency != Dataset.Column.ValueFrequency.Auto and \
                        result != value_frequency:
                    return Dataset.Column.ValueFrequency.Auto

                result = value_frequency

            return result

        # noinspection PyArgumentList
        def __new__(cls,
                    name: str,
                    supported: Iterable[Optional[Hashable]],
                    value_frequency: ValueFrequency = ValueFrequency.Auto):

            supported = set(supported)
            if len(supported) == 0 or next(iter(supported)) is None:
                contains_non_none = False

                for value in supported:
                    if value is not None:
                        contains_non_none = True
                        break

                if not contains_non_none:
                    return super(Dataset.Column, cls).__new__(cls, {None})

            elif value_frequency is not Dataset.Column.ValueFrequency.Discrete:
                types = set()
                for value in supported:
                    types.add(type(value) if not isinstance(value, type) else value)

                if value_frequency is Dataset.Column.ValueFrequency.Continuous:
                    supported = types
                elif value_frequency is Dataset.Column.ValueFrequency.Auto:
                    first_type = next(iter(types))
                    if len(types) == 1 and first_type is not str:
                        supported = {first_type}
            else:
                to_remove = set()

                for value in supported:
                    if isinstance(value, type):
                        to_remove.add(value)

                supported -= to_remove

            return super(Dataset.Column, cls).__new__(cls, supported)

        # noinspection PyUnusedLocal
        def __init__(self,
                     name: str,
                     supported: Iterable[Optional[Hashable]],
                     value_frequency: ValueFrequency = ValueFrequency.Auto):
            super().__init__()

            self._name: str = name

            if self.is_nil:
                value_frequency = Dataset.Column.ValueFrequency.Discrete
            self._value_frequency: Dataset.Column.ValueFrequency = value_frequency

        @property
        def is_nil(self) -> bool:
            return len(self) == 1 and None in self

        @property
        def name(self) -> str:
            return self._name

        @property
        def value_frequency(self) -> Dataset.Column.ValueFrequency:
            return self._value_frequency

        @property
        def supported(self) -> FrozenSet[Optional[Hashable]]:
            return frozenset(self)

        def supports_value_or_type(self, value: Optional[Hashable]) -> bool:
            return value in self or type(value) in self

        @property
        def values(self) -> FrozenSet[Optional[Hashable]]:
            return self.supported

        class UnsupportedValueError(ValueError):
            def __init__(self, column: Dataset.Column, value: Any):
                super().__init__(f"Unsupported value ({repr(value)}) for feature_column ({repr(column)}).")

        def __eq__(self, other: Dataset.Column) -> bool:
            return self._name == other._name

        def __hash__(self) -> int:
            return hash(self._name)

        string_value_separator: str = ':'

        def __str__(self) -> str:
            return f'{self._name}'

        def __repr__(self) -> str:
            return f'{self._name}{self.string_value_separator} {list(sorted(self, key=lambda value: str(value)))}'

        @classmethod
        def parse(cls, string_value: str, value_type: ValueFrequency = ValueFrequency.Auto) -> Dataset.Column:
            """
            >>> Dataset.Column.parse("A: (0, '1')")
            A: [0, '1']

            >>> Dataset.Column.parse("A: {0, 1}")
            A: [<class 'int'>]

            >>> Dataset.Column.parse("A: []")
            A: [None]

            >>> Dataset.Column.parse("A: [None, None]")
            A: [None]

            >>> Dataset.Column.parse("A")
            A: [None]

            >>> Dataset.Column("A", [int, str, 1], Dataset.Column.ValueFrequency.Discrete)
            A: [1]

            >>> Dataset.Column("A", [int, str, 1], Dataset.Column.ValueFrequency.Continuous)
            A: [<class 'int'>, <class 'str'>]

            :param string_value: name of the feature_column and
            the string representation of a Python collection separated by a colon (': ')
            :param value_type: use discrete, continuous or decide automatically which value type to use

            :return: parsed feature_column with the inferred possible value types.
            """

            split = string_value.strip().split(cls.string_value_separator, 1)

            name = split[0].strip()

            if len(split) == 2:
                feature_values = safe_literal_eval(split[1].strip())
            else:
                feature_values = {}

            return Dataset.Column(name, feature_values, value_type)

    class Row(Mapping["Dataset.Column", Hashable], Hashable):
        """
        Represents a row of a :class:`dataset<dataset.Dataset>` and stores its row as a mapping of
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

            string_value_separator = '->'

            def __str__(self) -> str:
                return f"{self._value}"

            def __repr__(self) -> str:
                return f"{str(self._column)} {self.string_value_separator} {repr(self._value)}"

            @classmethod
            def parse(cls, string_value: str) -> Dataset.Row.ColumnValue[Dataset.Column]:
                """
                >>> Dataset.Row.ColumnValue.parse("A: (0, '1') -> '1'")
                A -> '1'

                >>> Dataset.Row.ColumnValue.parse("A -> 1")
                A -> 1

                :param string_value: feature_column string representation with possible omitted supported row and
                the string of the value separated by a right arrow (' -> ')
                :return: parsed feature_column value with inferred possible feature_column value types and
                inferred value type

                :raises UnsupportedValueError: when the value isn't one of the supported feature_column row or
                doesn't match one of the supported types.
                """

                split = string_value.strip().split(cls.string_value_separator)

                value = safe_literal_eval(split[1].strip())

                if Dataset.Column.string_value_separator in split[0]:
                    column = Dataset.Column.parse(split[0].strip())
                else:
                    column = Dataset.Column(split[0].strip(), {value})

                return Dataset.Row.ColumnValue(column, value)

        def __init__(self,
                     values: Optional[
                         Union[Sequence[Dataset.Row.ColumnValue], Dict[Dataset.Column, Hashable]]] = None):
            if not isinstance(values, Dict):
                values = dict((column_value.column, column_value.value) for column_value in values)

            self._columns_to_values: Dict[Dataset.Column, Hashable] = values

            self._column_value_frequency: Optional[Dataset.Column.ValueFrequency] = \
                Dataset.Column.aggregate_value_frequencies(
                    column.value_frequency for column in self._columns_to_values.keys())

            for column, value in self._columns_to_values.items():
                if not column.supports_value_or_type(value):
                    raise Dataset.Column.UnsupportedValueError(column, value)

        @property
        def is_empty(self) -> bool:
            for value in self._columns_to_values.values():
                if value is not None:
                    return False

            return True

        @property
        def is_nil(self) -> bool:
            return len(self) == 0

        @property
        def column_value_frequency(self) -> Optional[Dataset.Column.ValueFrequency]:
            return self._column_value_frequency

        @property
        def columns(self) -> FrozenSet[Dataset.Column]:
            return frozenset(column for column in self._columns_to_values.keys())

        @property
        def column_tuple(self) -> Tuple[Dataset.Column, ...]:
            return tuple(column for column in self._columns_to_values.keys())

        @property
        def values(self) -> Tuple[Any, ...]:
            return tuple(value for value in self._columns_to_values.values())

        @property
        def column_values(self) -> FrozenSet[Dataset.Row.ColumnValue]:
            return frozenset(Dataset.Row.ColumnValue(column, value)
                             for column, value in self._columns_to_values.items())

        @property
        def column_values_tuple(self) -> Tuple[Dataset.Row.ColumnValue, ...]:
            return tuple(Dataset.Row.ColumnValue(column, value)
                         for column, value in self._columns_to_values.items())

        def get_value_for(self, column: Dataset.Column) -> Hashable:
            return self._columns_to_values[column]

        def __getitem__(self, column: Dataset.Column) -> Hashable:
            return self.get_value_for(column)

        def __len__(self) -> int:
            return len(self._columns_to_values)

        def __iter__(self) -> Iterator[Dataset.Column]:
            return iter(self._columns_to_values.keys())

        @property
        def feature_columns(self) -> FrozenSet[Dataset.Column]:
            return frozenset(self.column_tuple[:-1])

        @property
        def feature_column_tuple(self) -> Tuple[Dataset.Column, ...]:
            return self.column_tuple[:-1]

        @property
        def feature_values(self) -> Tuple[Any, ...]:
            return self.values[:-1]

        @property
        def feature_column_values(self) -> FrozenSet[Dataset.Row.ColumnValue]:
            return frozenset(self.column_values_tuple[:-1])

        @property
        def feature_column_value_tuple(self) -> Tuple[Dataset.Row.ColumnValue, ...]:
            return self.column_values_tuple[:-1]

        @property
        def feature_row(self) -> Dataset.Row:
            return Dataset.Row(self.feature_column_value_tuple)

        @property
        def result_column(self) -> Dataset.Column:
            return self.column_tuple[-1]

        @property
        def result_value(self) -> Any:
            return self.values[-1]

        @property
        def result_column_value(self) -> Dataset.Row.ColumnValue:
            return self.column_values_tuple[-1]

        @property
        def result_row(self) -> Dataset.Row:
            return Dataset.Row([self.result_column_value])

        def select(self, *columns: Dataset.Column) -> Dataset.Row:
            to_select = set(columns).intersection(self._columns_to_values.keys())
            return Dataset.Row([Dataset.Row.ColumnValue(column, self[column]) for column in to_select])

        def discard(self, *columns: Dataset.Column) -> Dataset.Row:
            return self.select(*set(self._columns_to_values.keys()).difference(columns))

        def __eq__(self, other: Dataset.Row) -> bool:
            return self._columns_to_values == other._columns_to_values

        def __hash__(self) -> int:
            return sum(hash(value) for value in self._columns_to_values.values())

        column_separator = ','

        def __str__(self) -> str:
            return self.column_separator.join(str(value) for value in self._columns_to_values.values())

        def __repr__(self) -> str:
            return f'{self.column_separator} '.join(
                f'{str(column)}: {repr(value)}' for column, value in self._columns_to_values.items())

        class InvalidRowValuesError(ValueError):
            _column_separator = '\n\t'

            def __init__(self, columns: List[Dataset.Column], values: List[Any]):
                super().__init__(f"Unsupported row ({repr(values)}) for columns:"
                                 f"{self._column_separator}"
                                 f"{self._column_separator.join(repr(column) for column in columns)}")

        @classmethod
        def parse(cls, columns: Sequence[Dataset.Column], string_value: str) -> Dataset.Row:
            """
            >>> Dataset.Row.parse([
            ...     Dataset.Column.parse("A: (0, '1')"),
            ...     Dataset.Column.parse("B: (3, 'six', None)")],
            ...     "0, six")
            A: 0, B: 'six'

            >>> Dataset.Row.parse([
            ...     Dataset.Column.parse("A: (0, '1')"),
            ...     Dataset.Column.parse("B: (3, 'six', None)")],
            ...     "'1'")
            A: '1', B: None


            :param columns: columns that the row of this row will represent.
            :param string_value: row of the row separated with a comma and space (', ')

            :return: parsed row with inferred value types

            :raises InvalidRowValuesError: when the row aren't one of the supported feature_column row or
            when they don't match the supported feature_column types.
            """
            if not isinstance(columns, list):
                columns = list(columns)

            values: List[Optional[str]] = string_value.strip().split(cls.column_separator)
            if len(columns) < len(values):
                raise Dataset.Row.InvalidRowValuesError(columns, values)
            if len(columns) > len(values):
                for column in columns[len(values):]:
                    if not column.supports_value_or_type(None):
                        raise Dataset.Row.InvalidRowValuesError(columns, values)

                values.extend([None] * (len(columns) - len(values)))

            try:
                return Dataset.Row(dict((column, (safe_literal_eval(value.strip())) if value is not None else None)
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

    # noinspection PyUnusedLocal
    def __init__(self, name: str,  rows: Iterable[Dataset.Row], columns: Optional[Sequence[Dataset.Column]] = None):
        super().__init__()
        self._name: str = name

        if isinstance(rows, Sequence):
            self._rows: Dict[Dataset.Row, None] = dict.fromkeys(rows)
        else:
            if not isinstance(rows, Sequence):
                rows = set(rows)
            self._rows: Set[Dataset.Row] = rows

        # Using dictionary keys for keeping things in order.
        # I might add an ordered set extension for this.
        if columns is not None:
            columns = dict.fromkeys(columns)
        else:
            columns = dict.fromkeys(next(iter(self)).columns)

        self._columns: Dict[Dataset.Column, None] = columns

        self._column_value_frequency: Optional[Dataset.Column.ValueFrequency] = \
            Dataset.Column.aggregate_value_frequencies(
                column.value_frequency for column in self._columns)

        column_set = self.columns
        for row in self:
            if row.columns != column_set:
                raise Dataset.NonMatchingRowsAndColumnsError(self._columns, self)

    @property
    def name(self) -> str:
        return self._name

    def is_empty(self) -> bool:
        return len(self) == 0

    def is_nil(self) -> bool:
        return len(self._columns.keys()) == 0

    @property
    def column_value_frequency(self) -> Optional[Dataset.Column.ValueFrequency]:
        return self._column_value_frequency

    @property
    def columns(self) -> FrozenSet[Dataset.Column]:
        return frozenset(self._columns.keys())

    @property
    def column_tuple(self) -> Tuple[Dataset.Column, ...]:
        return tuple(self._columns.keys())

    def get_column_named(self, name: str) -> Dataset.Column:
        for column in self._columns:
            if column.name == name:
                return column

        raise KeyError(f"No column named ({name}) in dataset ({str(self)})")

    @property
    def rows(self) -> FrozenSet[Dataset.Row]:
        return frozenset(self)

    @property
    def feature_columns(self) -> FrozenSet[Dataset.Column]:
        return frozenset(self.feature_column_tuple)

    @property
    def feature_column_tuple(self) -> Tuple[Dataset.Column, ...]:
        """
        >>> Dataset.parse("Test", "A, B \\n a, b").feature_column_tuple
        (A: ['a'],)

        :return: all but the last feature_column in the dataset, which represent features of the dataset
        """

        return self.column_tuple[:-1]

    @property
    def feature_rows(self) -> FrozenSet[Dataset.Row]:
        return frozenset(row.feature_row for row in self.rows)

    @property
    def feature_set(self) -> Dataset:
        return Dataset(self._name, self.feature_rows, self.feature_column_tuple)

    @property
    def result_column(self) -> Dataset.Column:
        """
        >>> Dataset.parse("Test", "A, B \\n a, b").result_column
        B: ['b']

        :return: the last feature_column of the dataset representing results of rows in the dataset
        """

        return list(self._columns.keys())[-1]

    @property
    def result_rows(self) -> FrozenSet[Dataset.Row]:
        return frozenset(row.result_row for row in self)

    @property
    def result_set(self) -> Dataset:
        return Dataset(self._name, self.result_rows, (self.result_column, ))

    def select(self, *columns: Dataset.Column) -> Dataset:
        """
        >>> Dataset.parse("Test", "A, B \\n a, b").select(Dataset.Column.parse("A"))
        A: ['a']
        A: 'a'

        :param columns: columns to select
        :return: a dataset with the selected columns
        """
        if len(columns) > 0:
            to_select = set(columns).intersection(self._columns.keys())
            return Dataset(self._name, (row.select(*to_select) for row in self), list(to_select))
        else:
            return Dataset(self._name, self, list(self._columns.keys()))

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[Dataset.Row]:
        return iter(self._rows)

    def __contains__(self, obj: object) -> bool:
        if isinstance(obj, Dataset.Row):
            return obj in self._rows

        return False

    def discard(self, *columns: Dataset.Column) -> Dataset:
        """
        >>> Dataset.parse("Test", "A, B \\n a, b").discard(Dataset.Column.parse("A"))
        B: ['b']
        B: 'b'

        :param columns: columns to discard
        :return: a dataset with the discarded columns
        """
        return self.select(*set(self._columns).difference(columns))

    def where(self, column_value: Dataset.Row.ColumnValue) -> Dataset:
        """
        >>> Dataset.parse("Test", "A, B \\n 1, b \\n c, b").where(Dataset.Row.ColumnValue.parse("A -> 'c'"))
        A: [1, 'c']
        B: ['b']
        A: 'c', B: 'b'

        :param column_value: feature_column value with which the rows will be filtered in
        :return: a dataset with the filtered rows
        """
        return Dataset(
            self._name,
            (row for row in self if row[column_value.column] == column_value.value),
            list(self._columns.keys()))

    def column_value_counts(self, column: Dataset.Column) -> Dict[Any, int]:
        """
        >>> Dataset.parse("Test", "A, B \\n a, b \\n c, b").column_value_counts(Dataset.Column.parse("B"))
        {'b': 2}

        :param column: the feature_column on which the feature_column counts are needed
        :return: dictionary of feature_column row to their count
        """
        result = dict()
        for row in self:
            value = row[column]
            if value in result:
                result[value] += 1
            else:
                result[value] = 1

        return result

    column_separator = ','
    row_separator = '\n'

    def __eq__(self, other: Dataset) -> bool:
        if self._columns != other._columns:
            return False

        return super().__eq__(other)

    def __str__(self) -> str:
        return f"{self.column_separator.join(str(column) for column in self._columns.keys())}" \
               f"{self.row_separator}" \
               f"{self.row_separator.join(str(row) for row in sorted(self, key=lambda row: str(row)))}"

    def __repr__(self) -> str:
        return f"{self.row_separator.join(repr(column) for column in self._columns.keys())}" \
               f"{self.row_separator}" \
               f"{self.row_separator.join(repr(row) for row in sorted(self, key=lambda row: str(row)))}"

    @classmethod
    def parse(cls,
              name: str,
              string_value: str,
              *column_value_frequencies: Dataset.Column.ValueFrequency) -> Dataset:
        """
        >>> Dataset.parse("My Dataset", "A, B \\n a, b")
        A: ['a']
        B: ['b']
        A: 'a', B: 'b'

        >>> Dataset.parse("My Dataset", "A, B \\n 3.4")
        A: [<class 'float'>]
        B: [None]
        A: 3.4, B: None

        >>> Dataset.parse("My Dataset", "A, B, C \\n 3.4, 1, a",
        ...     Dataset.Column.ValueFrequency.Continuous,
        ...     Dataset.Column.ValueFrequency.Discrete,
        ...     Dataset.Column.ValueFrequency.Continuous)
        A: [<class 'float'>]
        B: [1]
        C: [<class 'str'>]
        A: 3.4, B: 1, C: 'a'

        :param name: name of the dataset
        :param string_value: rows with row separated by a comma (', ') with
        the first row representing the feature_column names.

        :return: parsed dataset

        :raises InvalidRowValuesError: if a row is not the same length as the feature_column count
        """

        string_value = string_value.strip()
        lines_for_columns: List[List[str]] = []
        for line in string_value.split(cls.row_separator):
            lines_for_columns.append([value.strip() for value in line.strip().split(cls.column_separator)])

        columns = []
        if len(column_value_frequencies) == 0:
            column_value_frequencies = tuple(Dataset.Column.ValueFrequency.Auto for _ in lines_for_columns[0])
        elif len(column_value_frequencies) != len(lines_for_columns[0]):
            column_value_frequencies = tuple(column_value_frequencies[0] for _ in lines_for_columns[0])
        for index, column_name in enumerate(lines_for_columns[0]):
            columns.append(Dataset.Column(
                column_name,
                set((safe_literal_eval(row[index]) if index < len(row) else None) for row in lines_for_columns[1:]),
                column_value_frequencies[index]))

        lines_for_rows = [line.strip() for line in string_value.split(cls.row_separator)[1:]]
        rows = set()
        for line in lines_for_rows:
            rows.add(Dataset.Row.parse(columns, line))

        return Dataset(name, rows, columns)

    _base_test_file_path = os.path.join(".", "datasets")
    test_file_path = os.path.join(_base_test_file_path, "logic_small.csv")

    class TestFilePathPicker(Enum):
        Logic = 'logic'
        Titanic = 'Titanic'
        Volleyball = 'volleyball'

    test_file_paths = \
        {
            TestFilePathPicker.Logic: os.path.join(_base_test_file_path, "logic_small.csv"),
            TestFilePathPicker.Titanic: os.path.join(_base_test_file_path, "titanic_train_categorical.csv"),
            TestFilePathPicker.Volleyball: os.path.join(_base_test_file_path, "volleyball.csv")
        }

    @classmethod
    def from_csv_file(cls,
                      path_to_csv_file: str,
                      *column_value_frequencies: Dataset.Column.ValueFrequency) -> Dataset:
        # noinspection PyUnresolvedReferences
        """
        Dummy tests that should always work, but they are there for debugging purposes
        >>> os.path.exists(Dataset.test_file_path)
        True

        >>> set(os.path.exists(path) for path in Dataset.test_file_paths.values()) == {True}
        True

        >>> len(repr(Dataset.from_csv_file(Dataset.test_file_path))) > 0
        True


        :param path_to_csv_file: path to csv file
        :return: parsed dataset from file, as documented in the parse method
        """

        with open(path_to_csv_file) as csv_file:
            return cls.parse(os.path.basename(path_to_csv_file), csv_file.read(), *column_value_frequencies)
