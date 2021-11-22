from __future__ import annotations

import os
from enum import *
from typing import *

from extensions import safe_literal_eval, get_first, get_last, flatten, get


class Dataset(Sequence["Dataset.Row"]):
    """
    Represents a dataset used for machine learning purposes.
    It has interoperability with csv files and acts a bit as an SQL table with its columns and rows.
    """

    class Column(Collection[Optional[Hashable]], Hashable):
        """
        Represents a :class:`feature_column<dataset.Dataset.Column>` of a :class:`dataset<dataset.Dataset>` and
        stores the supported row and/or types of that column.
        If the value frequency is discrete it will store the provided supported row for
        the :class:`feature_column<dataset.Dataset.Column>`, but if the value frequency is continuous
        it will store the types of the provided supported row.
        If the value frequency is set to automatic or left default
        it will store the type of the provided supported row if there is only one type of supported row and
        if that type is not a string.
        """

        class ValueFrequency(Enum):
            Auto = 'a'
            Discrete = 'd'
            Continuous = 'c'

        @staticmethod
        def aggregate_value_frequencies(*value_frequencies: Union[Iterable, ValueFrequency]) -> \
                Optional[Dataset.Column.ValueFrequency]:
            result = None

            for value_frequency in flatten(value_frequencies):
                if result is not None and \
                        value_frequency != Dataset.Column.ValueFrequency.Auto and \
                        result != value_frequency:
                    return Dataset.Column.ValueFrequency.Auto

                result = value_frequency

            return result

        class InvalidValueFrequencyString(ValueError):
            def __init__(self, string_value: str):
                super().__init__(f"String ({string_value}) cannot be parsed into a value frequency.")

        @staticmethod
        def parse_value_frequency(string_value: str) -> Dataset.Column.ValueFrequency:
            if string_value == Dataset.Column.ValueFrequency.Discrete.value:
                return Dataset.Column.ValueFrequency.Discrete
            if string_value == Dataset.Column.ValueFrequency.Continuous.value:
                return Dataset.Column.ValueFrequency.Continuous
            if string_value == Dataset.Column.ValueFrequency.Auto.value:
                return Dataset.Column.ValueFrequency.Auto

            raise Dataset.Column.InvalidValueFrequencyString(string_value)

        def __init__(self,
                     name: str,
                     *supported: Union[Optional[Hashable], Iterable],
                     value_frequency: ValueFrequency = ValueFrequency.Auto,
                     preserve_supported_order: bool = True):

            """
            >>> Dataset.Column("A", 1, 2, 3)
            A: [<class 'int'>]

            >>> Dataset.Column("A", [1, 2, [3]])
            A: [<class 'int'>]

            >>> Dataset.Column("A", 1, 2, 3,
            ...     value_frequency=Dataset.Column.ValueFrequency.Discrete)
            A: [1, 2, 3]

            >>> Dataset.Column("A", 1, 2, 3, str,
            ...     value_frequency=Dataset.Column.ValueFrequency.Continuous)
            A: [<class 'int'>, <class 'str'>]

            >>> Dataset.Column("A", 1, 2, 3, int,
            ...     value_frequency=Dataset.Column.ValueFrequency.Discrete)
            A: [1, 2, 3]

            >>> Dataset.Column("A", 1, 2, 3, None)
            A: [<class 'int'>, None]

            >>> Dataset.Column("A", 1, 2, 3, str, None,
            ...     value_frequency=Dataset.Column.ValueFrequency.Continuous)
            A: [<class 'int'>, <class 'str'>, None]

            >>> Dataset.Column("A")
            A: [None]

            >>> Dataset.Column("A", None, None, None)
            A: [None]

            :param name: column name
            :param supported: Values or types or that the column will support. None can be passed in as well.

            :param value_frequency: Discrete, continuous or automatic. If its discrete, the column will discard all
            types from supported. If its continuous the column will discard all values from supported. If its auto
            the column will infer the type of values from supported if there is only one type, otherwise it leaves
            supported as it is. Default is auto.

            :param preserve_supported_order: should the order of supported be preserved. Default is True.
            """

            self._name: str = name

            self._supported_order_preserved = preserve_supported_order

            supported = tuple(flatten(supported))
            self.__set_supported(supported)

            if len(self._supported) == 0 and get_first(self._supported) is None:
                contains_non_none = False

                for value in self._supported:
                    if value is not None:
                        contains_non_none = True
                        break

                if not contains_non_none:
                    self.__set_supported(None)

            elif value_frequency is not Dataset.Column.ValueFrequency.Discrete:
                types = list()
                for value in supported:
                    types.append(type(value) if not (isinstance(value, type) or value is None) else value)
                types = list(dict.fromkeys(types).keys())

                if value_frequency is Dataset.Column.ValueFrequency.Continuous:
                    self.__set_supported(types)

                elif value_frequency is Dataset.Column.ValueFrequency.Auto:
                    first_type = get_first(types)
                    if len(types) == 1 and first_type is not str:
                        self.__set_supported(first_type)

                    elif len(types) == 2 and None in types and str not in types:
                        self.__set_supported(first_type, None)

            else:
                if isinstance(self._supported, Dict):
                    for value in supported:
                        if isinstance(value, type):
                            del self._supported[value]
                else:
                    for value in supported:
                        if isinstance(value, type):
                            self._supported.remove(value)

            if self.is_nil:
                value_frequency = Dataset.Column.ValueFrequency.Discrete
            self._value_frequency: Dataset.Column.ValueFrequency = value_frequency

        def __set_supported(self, *supported: Union[Optional[Hashable], Iterable]) -> NoReturn:
            if self._supported_order_preserved:
                self._supported: Dict[Optional[Hashable], None] = dict.fromkeys(flatten(supported))
            else:
                self._supported: Set[Optional, Hashable] = set(flatten(supported))

        @property
        def name(self) -> str:
            return self._name

        @property
        def value_frequency(self) -> Dataset.Column.ValueFrequency:
            return self._value_frequency

        @property
        def supported_order_preserved(self) -> bool:
            return self._supported_order_preserved

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
        def parse(cls, string_value: str) -> Dataset.Column:
            """
            >>> Dataset.Column.parse("A: (0, '1')")
            A: [0, '1']

            >>> Dataset.Column.parse("A: {0, 1}")
            A: [<class 'int'>]

            >>> Dataset.Column.parse("A: a{0, 1}")
            A: [<class 'int'>]

            >>> Dataset.Column.parse("A: d{0, 1}")
            A: [0, 1]

            >>> Dataset.Column.parse("A: c{0, 1}")
            A: [<class 'int'>]

            >>> Dataset.Column.parse("A: []")
            A: [None]

            >>> Dataset.Column.parse("A: [None, None]")
            A: [None]

            >>> Dataset.Column.parse("A")
            A: [None]

            >>> Dataset.Column("A", int, str, 1, value_frequency=Dataset.Column.ValueFrequency.Discrete)
            A: [1]

            >>> Dataset.Column("A", int, str, 1, value_frequency=Dataset.Column.ValueFrequency.Continuous)
            A: [<class 'int'>, <class 'str'>]

            :param string_value: name of the feature_column and
            the string representation of a Python collection separated by a colon (': ')

            :return: parsed feature_column with the inferred possible value_or_type types.
            """

            split = string_value.strip().split(cls.string_value_separator, 1)

            name = split[0].strip()
            value_frequency = Dataset.Column.ValueFrequency.Auto

            if len(split) == 2:
                split[1] = split[1].strip()
                try:
                    value_frequency = Dataset.Column.parse_value_frequency(split[1][0])
                    feature_values = safe_literal_eval(split[1][1:].strip())
                except Dataset.Column.InvalidValueFrequencyString:
                    feature_values = safe_literal_eval(split[1].strip())

            else:
                feature_values = {}

            return Dataset.Column(name, *feature_values, value_frequency=value_frequency)

    class Row(Mapping["Dataset.Column", Hashable], Hashable):
        """
        Represents a row of a :class:`dataset<dataset.Dataset>` and stores its row as a mapping of
        :class:`feature_column<dataset.Dataset.Column>` to value_or_type.
        """

        class ColumnValue(Hashable):
            """
            Represents the value of a :class:`feature_column<dataset.Dataset.Column>`
            in a :class:`row<dataset.Dataset.Row>` of a :class:`dataset<dataset.Dataset>` and
            keeps a reference of that :class:`feature_column<dataset.Dataset.Column>`.
            """

            def __init__(self, column: Dataset.Column, value: Any):
                self._column: Dataset.Column = column
                self._value: Hashable = value

                if not self._column.supports(value):
                    raise Dataset.Column.UnsupportedValueError(self._column, value)

            @property
            def column(self) -> Dataset.Column:
                return self._column

            @property
            def value(self) -> Any:
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

                :param string_value: column string representation with possible omitted supported row and
                the string of the value_or_type separated by a right arrow (' -> ')
                :return: parsed column value with inferred possible column value types

                :raises UnsupportedValueError: when the value_or_type isn't one of the supported feature_column row or
                doesn't match one of the supported types.
                """

                split = string_value.strip().split(cls.string_value_separator)

                value = safe_literal_eval(split[1].strip())

                if Dataset.Column.string_value_separator in split[0]:
                    column = Dataset.Column.parse(split[0].strip())
                else:
                    column = Dataset.Column(split[0].strip(), value)

                return Dataset.Row.ColumnValue(column, value)

        def __init__(self, *values: Union[Dataset.Row.ColumnValue, Iterable]) -> None:
            """
            >>> Dataset.Row([Dataset.Row.ColumnValue.parse("A -> 1")])
            A: 1

            >>> Dataset.Row(Dataset.Row.ColumnValue.parse("A -> 2"), [Dataset.Row.ColumnValue.parse("B -> 3")])
            A: 2; B: 3

            >>> Dataset.Row([])
            NIL ROW

            >>> Dataset.Row()
            NIL ROW

            :param values: either an iterable of column values or a mapping of columns to values or None
            """

            values = dict((column_value.column, column_value.value) for column_value in flatten(values))

            self._columns_to_values: Dict[Dataset.Column, Any] = values

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
        def is_empty(self) -> bool:
            for value in self._columns_to_values.values():
                if value is not None:
                    return False

            return True

        @property
        def is_nil(self) -> bool:
            return len(self) == 0

        @property
        def columns(self) -> Generator[Dataset.Column, None, None]:
            yield from self._columns_to_values

        @property
        def column_values(self) -> Generator[Dataset.Row.ColumnValue, None, None]:
            for column, value in self._columns_to_values.items():
                yield Dataset.Row.ColumnValue(column, value)

        @property
        def values(self) -> Generator[Any, None, None]:
            yield from self._columns_to_values.values()

        def get_feature_columns(self) -> Tuple[Dataset.Column, ...]:
            return tuple(self._columns_to_values)[:-1]

        def get_feature_values(self) -> Tuple[Any, ...]:
            return tuple(self._columns_to_values.values())[:-1]

        def get_feature_column_values(self) -> Tuple[Dataset.Row.ColumnValue, ...]:
            return tuple(self.column_values)[:-1]

        def get_feature_row(self) -> Dataset.Row:
            return Dataset.Row(self.get_feature_column_values())

        @property
        def result_column(self) -> Dataset.Column:
            return get_last(self._columns_to_values, type_limits={Dataset.Column})

        @property
        def result_value(self) -> Hashable:
            return self[self.result_column]

        def get_result_column_value(self) -> Dataset.Row.ColumnValue:
            result_column = self.result_column
            return Dataset.Row.ColumnValue(result_column, self._columns_to_values[result_column])

        def get_result_row(self) -> Dataset.Row:
            return Dataset.Row(self.get_result_column_value())

        def select(self, *to_select: Union[Iterable, Dataset.Column]) -> Dataset.Row:
            """
            >>> Dataset.Row.parse("0; 1").select(Dataset.Column.parse("0: {0}"))
            0: 0

            >>> Dataset.Row.parse("0; 1").select(Dataset.Column.parse("3: {0}"))
            NIL ROW

            >>> Dataset.Row.parse("0; 1; 2").select(Dataset.Column.parse("0: {0}"), Dataset.Column.parse("2: {2}"))
            0: 0; 2: 2

            >>> Dataset.Row.parse("0; 1; 2").select([Dataset.Column.parse("0: {0}"), Dataset.Column.parse("2: {2}")])
            0: 0; 2: 2


            :param to_select: either an iterable of columns or columns to select
            :return: a row with the selected columns
            """

            selection_set = set(flatten(to_select, type_limits={Dataset.Column}))
            return Dataset.Row((Dataset.Row.ColumnValue(column, self[column])
                                for column in self._columns_to_values if column in selection_set))

        def discard(self, *to_discard: Union[Iterable, Dataset.Column]) -> Dataset.Row:
            """
            >>> Dataset.Row.parse("0; 1").discard(Dataset.Column.parse("0: {0}"))
            1: 1

            >>> Dataset.Row.parse("0; 1").discard(Dataset.Column.parse("0: {0}"), Dataset.Column.parse("1: {1}"))
            NIL ROW


            :param to_discard: either an iterable of columns or columns to discard
            :return: a row without the discarded columns
            """

            return self.select(set(self._columns_to_values.keys())
                               .difference(flatten(to_discard, type_limits={Dataset.Column})))

        @overload
        def __getitem__(self, index: int) -> Any: ...

        @overload
        def __getitem__(self, row_slice: slice) -> List[Any]: ...

        @overload
        def __getitem__(self, column: Dataset.Column) -> Any: ...

        @overload
        def __getitem__(self, columns: Iterable) -> Generator[Any, None, None]: ...

        def __getitem__(self, obj: Union[int, slice, Iterable, Dataset.Column]) -> \
                Union[Hashable, List[Any], Generator[Hashable, None, None]]:
            """
            >>> Dataset.Row.parse("0; 1")[0]
            0

            >>> Dataset.Row.parse("0; 1; 2")[1:]
            [1, 2]

            >>> tuple(Dataset.Row.parse("0; 1; 2")[Dataset.Column.parse("0: {0}"), Dataset.Column.parse("2: {2}")])
            (0, 2)

            >>> Dataset.Row.parse("0; 1")[Dataset.Column.parse("0: {0}")]
            0

            >>> tuple(Dataset.Row.parse("0; 1; 2")[Dataset.Column.parse("0: {0}"), [Dataset.Column.parse("2: {2}")]])
            (0, 2)

            :param obj: either and int, slice, an iterable of columns or columns frow which to select the values
            :return: values of the at the index. slice, or selected columns
            """

            if isinstance(obj, int):
                return get(self._columns_to_values.values(), at=obj, type_limits={Dataset.Column})
            if isinstance(obj, slice):
                return list(self._columns_to_values.values())[obj]
            if isinstance(obj, Dataset.Column):
                return self._columns_to_values[obj]

            selection_set = set(flatten(obj, type_limits={Dataset.Column}))
            return (self._columns_to_values[column] for column in self._columns_to_values if column in selection_set)

        def __len__(self) -> int:
            return len(self._columns_to_values)

        def __iter__(self) -> Iterator[Dataset.Column]:
            return iter(self._columns_to_values)

        def __eq__(self, obj: object) -> bool:
            if isinstance(obj, Dataset.Row):
                return self._columns_to_values == obj._columns_to_values

            return False

        def __hash__(self) -> int:
            return sum(hash(value) for value in self._columns_to_values.values())

        column_separator = ';'
        nil_row_indicator = "NIL ROW"

        def __str__(self) -> str:
            if self.is_nil:
                return self.nil_row_indicator

            return self.column_separator.join(str(value) for value in self._columns_to_values.values())

        def __repr__(self) -> str:
            if self.is_nil:
                return self.nil_row_indicator

            return f'{self.column_separator} '.join(
                f'{str(column)}: {repr(value)}' for column, value in self._columns_to_values.items())

        class InvalidRowValuesError(ValueError):
            """
            Represents a scenario where the values of a row don't match its columns.
            """

            _column_separator = '\n\t'

            def __init__(self, columns: Iterable[Dataset.Column], values: Iterable[Any]):
                super().__init__(f"Unsupported row ({repr(values)}) for columns:"
                                 f"{self._column_separator}"
                                 f"{self._column_separator.join(repr(column) for column in columns)}")

        @classmethod
        def parse(cls, string_value: str, *columns: Union[Iterable, Dataset.Column]) -> Dataset.Row:
            """
            >>> Dataset.Row.parse("")
            NIL ROW

            >>> Dataset.Row.parse("0; six")
            0: 0; 1: 'six'

            >>> Dataset.Row.parse("'1'",
            ...     Dataset.Column.parse("A: (0, '1')"), [Dataset.Column.parse("B: (3, 'six', None)")])
            A: '1'; B: None

            >>> Dataset.Row.parse("A -> 1")
            A: 1

            >>> Dataset.Row.parse("A -> 1; B -> 2")
            A: 1; B: 2

            >>> Dataset.Row.parse("A -> 1; B -> 2; c")
            A: 1; B: 2

            >>> Dataset.Row.parse("A: {1, '1'} -> 1")
            A: 1

            >>> Dataset.Row.parse("A: {1, '1'} -> 1; B: {2} -> 2")
            A: 1; B: 2

            :param string_value: row of the row separated with a comma and space (', ')
            :param columns: columns that the row of this row will represent.

            :return: Parsed row with inferred value types and row names if the columns aren't present. The row names
            will be strings of ints starting from 1 up to 1 - the number of rows.

            :raises InvalidRowValuesError: when the row values aren't one of the supported column values or
            when they don't match the supported column types.
            """

            if string_value == "":
                return Dataset.Row()

            if len(columns) > 0:
                def create_column_value()

                values: List[Optional[str]] = [safe_literal_eval(value.strip())
                                               for value in string_value.strip().split(cls.column_separator)]

                columns = tuple(flatten(columns, type_limits={Dataset.Column}))
                if len(columns) < len(values):
                    raise Dataset.Row.InvalidRowValuesError(columns, values)
                if len(columns) > len(values):
                    for column in columns[len(values):]:
                        if not column.supports(None):
                            raise Dataset.Row.InvalidRowValuesError(columns, values)
                    values.extend([None] * (len(columns) - len(values)))

                column_values =

            else:
                if Dataset.Row.ColumnValue.string_value_separator in string_value:
                    return Dataset.Row(Dataset.Row.ColumnValue.parse(column_value_string.strip())
                                       for column_value_string in string_value.strip().split(cls.column_separator))

                else:
                    def create_column_value(index: int, value: Any) -> Dataset.Row.ColumnValue:
                        return Dataset.Row.ColumnValue(Dataset.Column(str(index), value), value)

                    column_values = (create_column_value(index, value)
                                     for index, value in enumerate(string_value.strip().split(cls.column_separator)))

            try:
                return Dataset.Row(Dataset.Row.ColumnValue(column, value) for column, value in zip(columns, values))
            except Dataset.Column.UnsupportedValueError:
                raise Dataset.Row.InvalidRowValuesError(columns, values)

    class NonMatchingRowsAndColumnsError(ValueError):
        _column_row_separator = '\t\n'

        def __init__(self,
                     columns: Iterable[Dataset.Column],
                     rows: Iterable[Dataset.Row[Dataset.Column, Dataset.Row.ColumnValue[Dataset.Column]]]):
            super().__init__(f"\nRows:{self._column_row_separator}" +
                             self._column_row_separator.join(repr(row) for row in rows) +
                             f"\ndon't match the columns:{self._column_row_separator}" +
                             self._column_row_separator.join(repr(column) for column in columns))

    def __init__(self,
                 name: str,
                 *rows: Union[Iterable[Dataset.Row], Dataset.Row],
                 columns: Optional[Iterable[Dataset.Column]] = None) -> None:
        """
        >>> Dataset("A")
        NIL DATASET

        >>> Dataset("A", Dataset.Row.parse("A: d{0} -> 0"), columns=Dataset.Column.parse("A: d{0}"))
        A: [0]
        A: 0

        >>> Dataset("A", Dataset.Row.parse("A: d{0} -> 0"))
        A: [0]
        A: 0

        >>> Dataset("A", Dataset.Row.parse("A: d{0} -> 0; B: d{0} -> 0"),
        ...     [Dataset.Row.parse("A: d{0} -> 0; B: d{0} -> 0")])
        A: [0]
        B: [0]
        A: 0; B: 0
        A: 0; B: 0

        >>> Dataset("A", columns=Dataset.Column.parse("A"))
        A: [None]

        :param name: name of dataset
        :param rows: iterables of dataset rows or dataset rows
        :param columns: dataset columns
        """

        self._name: str = name

        self._rows: List[Dataset.Row] = list(flatten(rows, type_limits={Dataset.Row}))

        # Using dictionary keys for keeping things in order.
        # I might add an ordered set extension for this.
        if columns is not None:
            columns = dict.fromkeys(flatten(columns, type_limits={Dataset.Column}))
        else:
            first_row = get_first(self, type_limits={Dataset.Row})
            if first_row is not None:
                columns = dict.fromkeys(first_row.columns)
            else:
                columns = dict()

        self._columns: Dict[Dataset.Column, None] = columns

        self._column_value_frequency: Optional[Dataset.Column.ValueFrequency] = \
            Dataset.Column.aggregate_value_frequencies(
                column.value_frequency for column in self.columns)

        columns = tuple(self.columns)
        for row in self:
            if tuple(row.columns) != columns:
                raise Dataset.NonMatchingRowsAndColumnsError(columns, self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def column_value_frequency(self) -> Optional[Dataset.Column.ValueFrequency]:
        return self._column_value_frequency

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def is_nil(self) -> bool:
        return len(self._columns.keys()) == 0

    @property
    def columns(self) -> Generator[Dataset.Column, None, None]:
        yield from self._columns.keys()

    def get_column_named(self, name: str) -> Dataset.Column:
        for column in self._columns:
            if column.name == name:
                return column

        raise KeyError(f"No column named ({name}) in dataset ({str(self)})")

    @property
    def values(self) -> Generator[Generator[Hashable, None, None], None, None]:
        for row in self:
            yield row.values

    def get_feature_columns(self) -> Tuple[Dataset.Column, ...]:
        return tuple(self._columns)[:-1]

    def get_feature_rows(self) -> Generator[Dataset.Row, None, None]:
        for row in self:
            yield row.get_feature_row()

    def get_feature_set(self) -> Dataset:
        return Dataset(self._name, *self.get_feature_rows(), columns=self.get_feature_columns())

    def get_feature_values(self) -> Generator[Tuple[Any, ...], None]:
        for row in self:
            yield row.get_feature_values()

    @property
    def result_column(self) -> Optional[Dataset.Column]:
        return get_last(self._columns.keys(), type_limits={Dataset.Column})

    def _get_result_rows(self) -> Generator[Dataset.Row, None, None]:
        for row in self:
            yield row.get_result_row()

    def get_result_set(self) -> Dataset:
        return Dataset(self._name, *self._get_result_rows(), columns=[self.result_column])

    @property
    def result_values(self) -> Generator[Hashable, None, None]:
        for row in self:
            yield row.result_value

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
                           (row.select(*to_select) for row in self),
                           columns=(column for column in self.columns if column in to_select))
        else:
            return Dataset(self._name, self, columns=self._columns.keys())

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

        def row_matches_column_values(row: Dataset.Row, *column_values_: Dataset.Row.ColumnValue):
            for column_value in column_values_:
                try:
                    if row[column_value.column] != column_value.value:
                        return False
                except KeyError:
                    return False

            return True

        return Dataset(
            self._name,
            (row for row in self if row_matches_column_values(row, *column_values)),
            columns=self._columns.keys())

    def column_value_counts(self, column: Dataset.Column) -> Dict[Hashable, int]:
        """
        >>> Dataset.parse("Test", "A, B \\n a, b \\n c, b").column_value_counts(Dataset.Column.parse("B: {'b'}"))
        {'b': 2}

        :param column: the column on which the feature_column counts are needed
        :return: dictionary of values to their count
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
    def __getitem__(self, *column_names: str) -> Dataset: ...

    @overload
    def __getitem__(self, *column_value: Dataset.Row.ColumnValue) -> Dataset: ...

    def __getitem__(self, obj: Union[
        int, slice,
        Dataset.Column, Tuple[Dataset.Column, ...], str, Tuple[str, ...],
        Dataset.Row.ColumnValue, Tuple[Dataset.Row.ColumnValue, ...]]) -> \
            Union[Dataset.Row, Dataset]:
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

        >>> Dataset.parse("My Dataset", "A, B, C \\n 3.4, 1, a \\n 2.3, 2, b ")["A", "B"]
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
            return Dataset(f"{self._name}/sliced", self._rows[obj], columns=self._columns)
        elif isinstance(obj, Dataset.Column):
            return self.select(obj)
        elif isinstance(obj, Dataset.Row.ColumnValue):
            return self.where(obj)
        elif isinstance(obj, Iterable):
            if set(type(value) for value in obj) == {Dataset.Column}:
                return self.select(*obj)
            elif set(type(value) for value in obj) == {str}:
                return self.select(*(self.get_column_named(column_name) for column_name in obj))
            elif set(type(value) for value in obj) == {Dataset.Row.ColumnValue}:
                return self.where(*obj)

        raise KeyError(f"Unsupported key ({str(obj)}) of type ({str(type(obj))}) for dataset ({str(self)}).")

    def __len__(self) -> int:
        return len(self._rows)

    column_separator = ','
    row_separator = '\n'
    nil_indicator = 'NIL DATASET'

    def __eq__(self, other: Dataset) -> bool:
        if self._columns != other._columns:
            return False

        return self._rows == other._rows

    def __str__(self) -> str:
        if self.is_nil:
            return self.nil_indicator

        if self.is_empty:
            return self.column_separator.join(str(column) for column in self._columns)

        return self.column_separator.join(str(column) for column in self._columns) + self.row_separator + \
            self.row_separator.join(self.column_separator.join(str(value) for value in row.values) for row in self)

    def __repr__(self) -> str:
        if self.is_nil:
            return self.nil_indicator

        if self.is_empty:
            return self.row_separator.join(repr(column) for column in self._columns)

        return self.row_separator.join(repr(column) for column in self._columns) + self.row_separator + \
            self.row_separator.join(repr(row) for row in self)

    @classmethod
    def parse(cls,
              name: str,
              string_value: str,
              *column_value_frequencies: Dataset.Column.ValueFrequency,
              columns: Optional[Iterable[Dataset.Column]] = None) -> Dataset:
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
        :param columns: set the columns so the dataset doesn't have to infer them from the string

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

        if columns is None:
            columns = []
            for index, column_name in enumerate(lines_for_columns[0]):
                columns.append(Dataset.Column(
                    column_name,
                    ((safe_literal_eval(row[index]) if index < len(row) else None) for row in lines_for_columns[1:]),
                    value_frequency=column_value_frequencies[index]))

        rows = list()
        for line in (line.strip() for line in string_value.split(cls.row_separator)[1:]):
            split = line.split(cls.column_separator)

            def create_column_value(column: Dataset.Column, column_index: int) -> Dataset.Row.ColumnValue:
                try:
                    return Dataset.Row.ColumnValue(column, safe_literal_eval(split[column_index].strip()))
                except IndexError:
                    return Dataset.Row.ColumnValue(column, None)

            rows.append(Dataset.Row(create_column_value(column, index) for index, column in enumerate(columns)))

        return Dataset(name, *rows, columns=columns)

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
                      *column_value_frequencies: Dataset.Column.ValueFrequency,
                      columns: Optional[Sequence[Dataset.Column]] = None) -> Dataset:
        # noinspection PyUnresolvedReferences
        """
        >>> os.path.exists(Dataset.test_file_path)
        True

        >>> set(os.path.exists(path) for path in Dataset.test_file_paths.values()) == {True}
        True

        :param path_to_csv_file: path to csv file
        :param columns: set the columns so the dataset doesn't have to infer them from the file
        :return: parsed dataset from file, as documented in the parse method
        """

        with open(path_to_csv_file) as csv_file:
            return cls.parse(os.path.basename(path_to_csv_file),
                             csv_file.read(),
                             *column_value_frequencies,
                             columns=columns)
