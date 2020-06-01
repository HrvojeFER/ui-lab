from __future__ import annotations

import os
from enum import *
from typing import *

from extensions import safe_literal_eval


class Dataset(Sequence["Dataset.Row"]):
    """
    Represents a dataset used for machine learning purposes.
    It has interoperability with csv files and acts a bit as an SQL table with its columns and rows.
    """

    class Column(Collection[Optional[Hashable]], Hashable):
        """
        Represents a :class:`feature_column<dataset.Dataset.Column>` of a :class:`dataset<dataset.Dataset>` and
        stores the supported row and/or types of that feature_column.
        If the value_or_type frequency is discrete it will store the provided supported row for
        the :class:`feature_column<dataset.Dataset.Column>`, but if the value_or_type frequency is continuous
        it will store the types of the provided supported row.
        If the value_or_type frequency is set to automatic or left default
        it will store the type of the provided supported row if there is only one type of supported row and
        if that type is not a string.
        """

        class ValueFrequency(Enum):
            Auto = 'Auto'
            Discrete = 'Discrete'
            Continuous = 'Continuous'

        @staticmethod
        def aggregate_value_frequencies(value_frequencies: Iterable[Dataset.Column.ValueFrequency]) -> \
                Optional[Dataset.Column.ValueFrequency]:
            result = None

            for value_frequency in value_frequencies:
                if result is not None and \
                        value_frequency != Dataset.Column.ValueFrequency.Auto and \
                        result != value_frequency:
                    return Dataset.Column.ValueFrequency.Auto

                result = value_frequency

            return result

        def __init__(self,
                     name: str,
                     supported: Sequence[Optional[Hashable]],
                     value_frequency: ValueFrequency = ValueFrequency.Auto):

            self._supported: Dict[Optional[Hashable], None] = dict.fromkeys(
                value for value in supported if isinstance(value, Hashable))
            if len(self._supported) == 0 or next(iter(self._supported)) is None:
                contains_non_none = False

                for value in self._supported:
                    if value is not None:
                        contains_non_none = True
                        break

                if not contains_non_none:
                    self._supported = {None: None}

            elif value_frequency is not Dataset.Column.ValueFrequency.Discrete:
                types = set()
                for value in supported:
                    types.add(type(value) if not isinstance(value, type) else value)

                if value_frequency is Dataset.Column.ValueFrequency.Continuous:
                    self._supported = dict.fromkeys(types)
                elif value_frequency is Dataset.Column.ValueFrequency.Auto:
                    first_type = next(iter(types))
                    if len(types) == 1 and first_type is not str:
                        self._supported = {first_type: None}

            else:
                for value in supported:
                    if isinstance(value, type) and value in self._supported:
                        del self._supported[value]

            self._name: str = name

            if self.is_nil:
                value_frequency = Dataset.Column.ValueFrequency.Discrete
            self._value_frequency: Dataset.Column.ValueFrequency = value_frequency

        @property
        def name(self) -> str:
            return self._name

        @property
        def value_frequency(self) -> Dataset.Column.ValueFrequency:
            return self._value_frequency

        def supports(self, value_or_type: Optional[Hashable]) -> bool:
            return value_or_type in self or type(value_or_type) in self

        @property
        def is_nil(self) -> bool:
            return len(self) == 1 and None in self

        def __iter__(self) -> Iterator[Optional[Hashable]]:
            return iter(self._supported)

        def __len__(self) -> int:
            return len(self._supported)

        def __contains__(self, item: object) -> bool:
            return item in self._supported

        class UnsupportedValueError(ValueError):
            def __init__(self, column: Dataset.Column, value: Any):
                super().__init__(f"Unsupported value_or_type ({repr(value)}) for feature_column ({repr(column)}).")

        def __eq__(self, other: Dataset.Column) -> bool:
            return self._name == other._name and self._supported == other._supported

        def __hash__(self) -> int:
            return hash(self._name)

        string_value_separator: str = ':'

        def __str__(self) -> str:
            return f'{self._name}'

        def __repr__(self) -> str:
            return f'{self._name}{self.string_value_separator} {str(list(self._supported))}'

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
            :param value_type: use discrete, continuous or decide automatically which value_or_type type to use

            :return: parsed feature_column with the inferred possible value_or_type types.
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
        :class:`feature_column<dataset.Dataset.Column>` to value_or_type.
        """

        class ColumnValue(Hashable):
            """
            Represents the value_or_type of a :class:`feature_column<dataset.Dataset.Column>`
            in a :class:`row<dataset.Dataset.Row>` of a :class:`dataset<dataset.Dataset>` and
            keeps a reference of that :class:`feature_column<dataset.Dataset.Column>`.
            """

            def __init__(self, column: Dataset.Column, value: Hashable):
                self._column: Dataset.Column = column
                self._value: Hashable = value

                if not self._column.supports(value):
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
                the string of the value_or_type separated by a right arrow (' -> ')
                :return: parsed feature_column value_or_type with inferred possible feature_column value_or_type types
                and inferred value_or_type type

                :raises UnsupportedValueError: when the value_or_type isn't one of the supported feature_column row or
                doesn't match one of the supported types.
                """

                split = string_value.strip().split(cls.string_value_separator)

                value = safe_literal_eval(split[1].strip())

                if Dataset.Column.string_value_separator in split[0]:
                    column = Dataset.Column.parse(split[0].strip())
                else:
                    column = Dataset.Column(split[0].strip(), [value])

                return Dataset.Row.ColumnValue(column, value)

        def __init__(self,
                     values: Optional[Union[
                         Sequence[Dataset.Row.ColumnValue], Dict[Dataset.Column, Hashable]]] = None):
            if not isinstance(values, Dict):
                values = dict((column_value.column, column_value.value) for column_value in values)

            self._columns_to_values: Dict[Dataset.Column, Hashable] = values

            self._column_value_frequency: Optional[Dataset.Column.ValueFrequency] = \
                Dataset.Column.aggregate_value_frequencies(
                    column.value_frequency for column in self._columns_to_values.keys())

            for column, value in self._columns_to_values.items():
                if not column.supports(value):
                    raise Dataset.Column.UnsupportedValueError(column, value)

        @property
        def column_value_frequency(self) -> Optional[Dataset.Column.ValueFrequency]:
            return self._column_value_frequency

        @property
        def column_set(self) -> FrozenSet[Dataset.Column]:
            return frozenset(column for column in self._columns_to_values.keys())

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
        def column_tuple(self) -> Tuple[Dataset.Column, ...]:
            return tuple(column for column in self._columns_to_values.keys())

        @property
        def column_value_set(self) -> FrozenSet[Dataset.Row.ColumnValue]:
            return frozenset(Dataset.Row.ColumnValue(column, value)
                             for column, value in self._columns_to_values.items())

        @property
        def column_value_tuple(self) -> Tuple[Dataset.Row.ColumnValue, ...]:
            return tuple(Dataset.Row.ColumnValue(column, value)
                         for column, value in self._columns_to_values.items())

        def get_value_for(self, column: Dataset.Column) -> Hashable:
            return self._columns_to_values[column]

        @property
        def values(self) -> Tuple[Hashable, ...]:
            return tuple(value for value in self._columns_to_values.values())

        @property
        def feature_column_set(self) -> FrozenSet[Dataset.Column]:
            return frozenset(self.feature_column_tuple)

        @property
        def feature_column_tuple(self) -> Tuple[Dataset.Column, ...]:
            return self.column_tuple[:-1]

        @property
        def feature_values(self) -> Tuple[Hashable, ...]:
            return self.values[:-1]

        @property
        def feature_column_value_set(self) -> FrozenSet[Dataset.Row.ColumnValue]:
            return frozenset(self.column_value_tuple[:-1])

        @property
        def feature_column_value_tuple(self) -> Tuple[Dataset.Row.ColumnValue, ...]:
            return self.column_value_tuple[:-1]

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
            return self.column_value_tuple[-1]

        @property
        def result_row(self) -> Dataset.Row:
            return Dataset.Row([self.result_column_value])

        def select(self, *columns: Dataset.Column) -> Dataset.Row:
            to_select = set(columns).intersection(self._columns_to_values.keys())
            return Dataset.Row([Dataset.Row.ColumnValue(column, self[column])
                                for column in self._columns_to_values.keys() if column in to_select])

        def discard(self, *columns: Dataset.Column) -> Dataset.Row:
            return self.select(*set(self._columns_to_values.keys()).difference(columns))

        @overload
        def __getitem__(self, index: int) -> Hashable: ...

        @overload
        def __getitem__(self, column: Dataset.Column) -> Hashable: ...

        @overload
        def __getitem__(self, column_slice: slice) -> Tuple[Hashable, ...]: ...

        @overload
        def __getitem__(self, *columns: Dataset.Column) -> Dataset.Row: ...

        def __getitem__(self, obj: object) -> Union[Hashable, Tuple[Hashable, ...], Dataset.Row]:
            """
            >>> Dataset.Row.parse("0; 1")[0]
            0

            >>> Dataset.Row.parse("0; 1; 2")[:2]
            (0, 1)

            >>> Dataset.Row.parse("0; 1")[Dataset.Column.parse("0: {0}")]
            0

            >>> Dataset.Row.parse("0; 1; 2")[Dataset.Column.parse("0: {0}"), Dataset.Column.parse("2: {2}")]
            0: 0; 2: 2


            :param obj: object to slice the column with

            :return: If the object is of type int, returns the value in the column with that index. If the object
            is of type slice, returns a tuple of values that would get sliced as with other Python collections.
            If the object is of type column, returns the value at that column. If there are multiple objects and
            they are all of type column, returns a row with the selected columns - this is the same as calling
            select on this column with those columns.
            """

            if isinstance(obj, Dataset.Column):
                return self.get_value_for(obj)
            elif isinstance(obj, int) or isinstance(obj, slice):
                return self.values[obj]
            elif isinstance(obj, Dataset.Column):
                return self.select(obj)
            elif isinstance(obj, Iterable) and set(type(value) for value in obj) == {Dataset.Column}:
                return self.select(*obj)

            raise KeyError(f"Unsupported key ({str(obj)}) of type ({str(type(obj))}) for row ({str(self)}).")

        def __len__(self) -> int:
            return len(self._columns_to_values)

        def __iter__(self) -> Iterator[Dataset.Column]:
            return iter(self._columns_to_values.keys())

        def __eq__(self, obj: object) -> bool:
            if isinstance(obj, Dataset.Row):
                return self._columns_to_values == obj._columns_to_values

            return False

        def __hash__(self) -> int:
            return sum(hash(value) for value in self._columns_to_values.values())

        column_separator = ';'

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
        def parse(cls, string_value: str, columns: Optional[Sequence[Dataset.Column]] = None) -> Dataset.Row:
            """
            >>> Dataset.Row.parse("0; six")
            0: 0; 1: 'six'

            >>> Dataset.Row.parse("'1'",
            ...     [Dataset.Column.parse("A: (0, '1')"),
            ...      Dataset.Column.parse("B: (3, 'six', None)")])
            A: '1'; B: None

            >>> Dataset.Row.parse("A -> 1; B -> 2")
            A: 1; B: 2

            >>> Dataset.Row.parse("A: {1, '1'} -> 1; B: {2} -> 2")
            A: 1; B: 2

            :param string_value: row of the row separated with a comma and space (', ')
            :param columns: columns that the row of this row will represent.

            :return: Parsed row with inferred value types and row names if the columns aren't present. The row names
            will be strings of ints starting from 1 up to 1 - the number of rows.

            :raises InvalidRowValuesError: when the row aren't one of the supported feature_column row or
            when they don't match the supported feature_column types.
            """
            if columns is not None:
                values: List[Optional[str]] = [safe_literal_eval(value.strip())
                                               for value in string_value.strip().split(cls.column_separator)]

                if not isinstance(columns, list):
                    columns = list(columns)

                if len(columns) < len(values):
                    raise Dataset.Row.InvalidRowValuesError(columns, values)
                if len(columns) > len(values):
                    for column in columns[len(values):]:
                        if not column.supports(None):
                            raise Dataset.Row.InvalidRowValuesError(columns, values)
                    values.extend([None] * (len(columns) - len(values)))

            else:
                if Dataset.Row.ColumnValue.string_value_separator in string_value:
                    return Dataset.Row([Dataset.Row.ColumnValue.parse(column_value_string.strip())
                                        for column_value_string in string_value.strip().split(cls.column_separator)])

                else:
                    values: List[Optional[str]] = [safe_literal_eval(value.strip())
                                                   for value in string_value.strip().split(cls.column_separator)]
                    columns = [Dataset.Column(str(index), [value]) for index, value in enumerate(values)]

            try:
                return Dataset.Row(dict(((column, value) for column, value in zip(columns, values))))
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
    def __init__(self, name: str, rows: Sequence[Dataset.Row], columns: Optional[Sequence[Dataset.Column]] = None):
        self._name: str = name

        self._rows: List[Dataset.Row] = list(rows)

        # Using dictionary keys for keeping things in order.
        # I might add an ordered set extension for this.
        if columns is not None:
            columns = dict.fromkeys(columns)
        else:
            columns = dict.fromkeys(next(iter(self)).column_set)

        self._columns: Dict[Dataset.Column, None] = columns

        self._column_value_frequency: Optional[Dataset.Column.ValueFrequency] = \
            Dataset.Column.aggregate_value_frequencies(
                column.value_frequency for column in self._columns)

        column_set = self.column_set
        for row in self:
            if row.column_set != column_set:
                raise Dataset.NonMatchingRowsAndColumnsError(self._columns, self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def column_value_frequency(self) -> Optional[Dataset.Column.ValueFrequency]:
        return self._column_value_frequency

    def is_empty(self) -> bool:
        return len(self) == 0

    def is_nil(self) -> bool:
        return len(self._columns.keys()) == 0

    @property
    def column_set(self) -> FrozenSet[Dataset.Column]:
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
    def row_set(self) -> FrozenSet[Dataset.Row]:
        return frozenset(self)

    @property
    def values(self) -> Tuple[Tuple[Hashable, ...], ...]:
        return tuple(row.values for row in self)

    @property
    def feature_column_set(self) -> FrozenSet[Dataset.Column]:
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
    def feature_row_set(self) -> FrozenSet[Dataset.Row]:
        return frozenset(row.feature_row for row in self)

    @property
    def feature_row_tuple(self) -> Tuple[Dataset.Row, ...]:
        return tuple(row.feature_row for row in self)

    @property
    def feature_set(self) -> Dataset:
        return Dataset(self._name, self, self.column_tuple)

    def feature_values(self) -> Tuple[Tuple[Hashable, ...], ...]:
        return tuple(row.feature_values for row in self)

    @property
    def result_column(self) -> Dataset.Column:
        """
        >>> Dataset.parse("Test", "A, B \\n a, b").result_column
        B: ['b']

        :return: the last feature_column of the dataset representing results of rows in the dataset
        """

        return list(self._columns.keys())[-1]

    @property
    def result_row_set(self) -> FrozenSet[Dataset.Row]:
        return frozenset(row.result_row for row in self)

    @property
    def result_row_tuple(self) -> Tuple[Dataset.Row, ...]:
        return tuple(row.result_row for row in self)

    @property
    def result_set(self) -> Dataset:
        return Dataset(self._name, self.result_row_tuple, (self.result_column,))

    @property
    def result_values(self) -> Tuple[Hashable, ...]:
        return tuple(row.result_value for row in self)

    def select(self, *columns: Dataset.Column) -> Dataset:
        """
        >>> Dataset.parse("Test", "A, B \\n a, b").select(Dataset.Column.parse("A: {'a'}"))
        A: ['a']
        A: 'a'

        :param columns: columns to select
        :return: a dataset with the selected columns
        """
        if len(columns) > 0:
            to_select = set(columns).intersection(self._columns.keys())
            return Dataset(self._name,
                           [row.select(*to_select) for row in self],
                           [column for column in self.column_tuple if column in to_select])
        else:
            return Dataset(self._name, self, list(self._columns.keys()))

    def discard(self, *columns: Dataset.Column) -> Dataset:
        """
        >>> Dataset.parse("Test", "A, B \\n a, b").discard(Dataset.Column.parse("A: {'a'}"))
        B: ['b']
        B: 'b'

        :param columns: columns to discard
        :return: a dataset with the discarded columns
        """
        return self.select(*set(self._columns).difference(columns))

    def where(self, *column_values: Dataset.Row.ColumnValue) -> Dataset:
        """
        >>> Dataset.parse("Test", "A, B \\n 1, b \\n c, b").where(Dataset.Row.ColumnValue.parse("A: {1, 'c'} -> 'c'"))
        A: [1, 'c']
        B: ['b']
        A: 'c'; B: 'b'

        :param column_values: column values with which the rows will be filtered in
        :return: a dataset with the filtered rows
        """
        def __row_matches_column_values(row: Dataset.Row, *column_values_: Dataset.Row.ColumnValue):
            for column_value in column_values_:
                try:
                    if row[column_value.column] != column_value.value:
                        return False
                except KeyError:
                    return False

            return True

        return Dataset(
            self._name,
            [row for row in self if __row_matches_column_values(row, *column_values)],
            list(self._columns.keys()))

    def column_value_counts(self, column: Dataset.Column) -> Dict[Hashable, int]:
        """
        >>> Dataset.parse("Test", "A, B \\n a, b \\n c, b").column_value_counts(Dataset.Column.parse("B: {'b'}"))
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

    @overload
    def __getitem__(self, index: int) -> Dataset.Row: ...

    @overload
    def __getitem__(self, dataset_slice: slice) -> Dataset: ...

    @overload
    def __getitem__(self, *column: Dataset.Column) -> Dataset: ...

    @overload
    def __getitem__(self, *column_value: Dataset.Row.ColumnValue) -> Dataset: ...

    def __getitem__(self, obj: object) -> Union[Dataset.Row, Dataset]:
        """
        >>> Dataset.parse("My Dataset", "A, B, C \\n 3.4, 1, a")[0]
        A: 3.4; B: 1; C: 'a'

        >>> Dataset.parse("My Dataset", "A, B, C \\n 3.4, 1, a \\n 2.3, 2, b ")[1:]
        A: [<class 'float'>]
        B: [<class 'int'>]
        C: ['a', 'b']
        A: 2.3; B: 2; C: 'b'

        >>> Dataset.parse("My Dataset", "A, B, C \\n 3.4, 1, a \\n 2.3, 2, b ")[
        ...     Dataset.Column.parse("A: {3.4}"), Dataset.Column.parse("B: {2}")]
        A: [<class 'float'>]
        B: [<class 'int'>]
        A: 3.4; B: 1
        A: 2.3; B: 2

        >>> Dataset.parse("My Dataset", "A, B, C \\n 3.4, 1, a \\n 2.3, 2, b ")[
        ...     Dataset.Row.ColumnValue.parse("A -> 2.3")]
        A: [<class 'float'>]
        B: [<class 'int'>]
        C: ['a', 'b']
        A: 2.3; B: 2; C: 'b'


        :param obj: object to slice the dataset with

        :return: If the object is of type int, returns the row at the index represented by the object. If the object
        is of type slice, returns a dataset that would have sliced rows as with other Python collections.
        If the object is a tuple of columns, returns the dataset with the selected columns -
        this is the same as using the select method. If the object is a tuple of column values, returns a dataset with
        all the rows having those column values - this is the same as calling the where method.
        """

        if isinstance(obj, int):
            return self._rows[obj]
        elif isinstance(obj, slice):
            return Dataset(f"{self._name}/sliced", self._rows[obj], self._columns)
        elif isinstance(obj, Dataset.Column):
            return self.select(obj)
        elif isinstance(obj, Dataset.Row.ColumnValue):
            return self.where(obj)
        elif isinstance(obj, Iterable):
            if set(type(value) for value in obj) == {Dataset.Column}:
                return self.select(*obj)
            elif set(type(value) for value in obj) == {Dataset.Row.ColumnValue}:
                return self.where(*obj)

        raise KeyError(f"Unsupported key ({str(obj)}) of type ({str(type(obj))}) for dataset ({str(self)}).")

    def __len__(self) -> int:
        return len(self._rows)

    column_separator = ','
    row_separator = '\n'

    def __eq__(self, other: Dataset) -> bool:
        if self._columns != other._columns:
            return False

        return self._rows == other._rows

    def __str__(self) -> str:
        return f"{self.column_separator.join(str(column) for column in self._columns.keys())}" \
               f"{self.row_separator}" + \
               self.row_separator.join(
                   self.column_separator.join(str(value) for value in row.values) for row in self)

    def __repr__(self) -> str:
        return f"{self.row_separator.join(repr(column) for column in self._columns.keys())}" \
               f"{self.row_separator}" \
               f"{self.row_separator.join(repr(row) for row in self)}"

    @classmethod
    def parse(cls,
              name: str,
              string_value: str,
              *column_value_frequencies: Dataset.Column.ValueFrequency) -> Dataset:
        """
        >>> Dataset.parse("My Dataset", "A, B \\n a, b")
        A: ['a']
        B: ['b']
        A: 'a'; B: 'b'

        >>> Dataset.parse("My Dataset", "A, B \\n 3.4")
        A: [<class 'float'>]
        B: [None]
        A: 3.4; B: None

        >>> Dataset.parse("My Dataset", "A, B, C \\n 3.4, 1, a \\n 2.3, 2, b ",
        ...     Dataset.Column.ValueFrequency.Continuous,
        ...     Dataset.Column.ValueFrequency.Discrete,
        ...     Dataset.Column.ValueFrequency.Continuous)
        A: [<class 'float'>]
        B: [1, 2]
        C: [<class 'str'>]
        A: 3.4; B: 1; C: 'a'
        A: 2.3; B: 2; C: 'b'

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

        if len(column_value_frequencies) == 0:
            column_value_frequencies = tuple(Dataset.Column.ValueFrequency.Auto for _ in lines_for_columns[0])
        elif len(column_value_frequencies) != len(lines_for_columns[0]):
            column_value_frequencies = tuple(column_value_frequencies[0] for _ in lines_for_columns[0])

        columns = []
        for index, column_name in enumerate(lines_for_columns[0]):
            columns.append(Dataset.Column(
                column_name,
                [(safe_literal_eval(row[index]) if index < len(row) else None) for row in lines_for_columns[1:]],
                column_value_frequencies[index]))

        lines_for_rows = [line.strip() for line in string_value.split(cls.row_separator)[1:]]

        rows = list()
        for line in lines_for_rows:
            rows.append(Dataset.Row.parse(
                Dataset.Row.column_separator.join(line.split(cls.column_separator)), columns))

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
        >>> os.path.exists(Dataset.test_file_path)
        True

        >>> set(os.path.exists(path) for path in Dataset.test_file_paths.values()) == {True}
        True

        :param path_to_csv_file: path to csv file
        :return: parsed dataset from file, as documented in the parse method
        """

        with open(path_to_csv_file) as csv_file:
            return cls.parse(os.path.basename(path_to_csv_file), csv_file.read(), *column_value_frequencies)
