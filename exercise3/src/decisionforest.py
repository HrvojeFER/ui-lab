from __future__ import annotations

from typing import *

from dataset import Dataset
from extensions import max_argument, safe_literal_eval
from entropy import information_gain


class DecisionForest(FrozenSet):
    """
    Represents a decision :class:`forest<decisionforest.DecisionForest.Tree>` used for
    classifying data samples.
    """

    class Tree(Hashable):
        """
        Represents a decision :class:`tree<decisionforest.DecisionForest.Tree>` used for
        classifying data samples.
        """
    
        class Node(Hashable, Mapping[Any, "DecisionForest.Tree.Node"]):
            """
            Represents a :class:`node<decisionforest.DecisionForest.Tree.Node>` in
            a decision :class:`tree<decisionforest.DecisionForest.Tree>`.
            """
            def __init__(self,
                         feature_column: Dataset.Column,
                         connections_or_value: Union[
                             Union[Iterable[DecisionForest.Tree.Connection],
                                   Mapping[Any, DecisionForest.Tree.Node]],
                             Any] = None):
                self._feature_column: Dataset.Column = feature_column

                if isinstance(connections_or_value, Mapping):
                    connections_or_value = dict(connections_or_value)
                    self._connections: Dict[Any, DecisionForest.Tree.Node] = connections_or_value
                elif isinstance(connections_or_value, Iterable) and not isinstance(connections_or_value, str):
                    connections_or_value = dict((connection.feature_value, connection.node)
                                                for connection in connections_or_value)
                    self._connections: Dict[Any, DecisionForest.Tree.Node] = connections_or_value
                else:
                    self._connections: Dict[Any, DecisionForest.Tree.Node] = dict()
                    self._value: Any = connections_or_value

                for connection in self.connections:
                    if not feature_column.supports(connection.feature_value):
                        raise Dataset.Column.UnsupportedValueError(feature_column, connection.feature_value)
    
            @property
            def feature_column(self) -> Dataset.Column:
                return self._feature_column
    
            @property
            def connections(self) -> FrozenSet[DecisionForest.Tree.Connection]:
                return frozenset(DecisionForest.Tree.Connection(feature_value, node)
                                 for feature_value, node in self._connections.items())

            @property
            def connection_tuple(self) -> Tuple[DecisionForest.Tree.Connection, ...]:
                return tuple(DecisionForest.Tree.Connection(feature_value, node)
                             for feature_value, node in self._connections.items())

            def is_leaf(self) -> bool:
                return len(self) == 0

            @property
            def value(self) -> Optional[Any]:
                if self.is_leaf():
                    return self._value
    
            def __getitem__(self, feature_value: Any) -> DecisionForest.Tree.Node:
                return self._connections[feature_value]
    
            def __len__(self) -> int:
                return len(self._connections)
    
            def __iter__(self) -> Iterator[Any]:
                return iter(self._connections)
    
            class ConnectionNotFound(ValueError):
                def __init__(self, node: DecisionForest.Tree.Node, *feature_values: Any):
                    super().__init__(f"No connection for feature row ({str([*feature_values])}) found in the "
                                     f"nodes ({str(node)}) connections_or_value ({str(node.connections)}).")

            def iterate(self, row: Dataset.Row) -> DecisionForest.Tree.Path:
                """
                >>> node = DecisionForest.Tree.Node.parse("B =>(c --> D =>(True --> S -> 0); d --> A -> 1)")
                >>> str(node.iterate(Dataset.Row.parse("B: {'c', 'd'} -> d")))
                'B ==> A'

                >>> str(node.iterate(Dataset.Row.parse("B: {'c', 'd'} -> c")))
                'B ==> D'

                >>> str(node.iterate(Dataset.Row.parse("B: {'c', 'd'} -> c; D -> True")))
                'B ==> D ==> S'


                :param row: feature row with which to iterate over the nodes

                :return: path the nodes took

                :raises ConnectionNotFound when the connection for the feature value_or_type cannot be found
                """

                nodes = [self]

                while True:
                    try:
                        if nodes[-1].is_leaf() or row.is_nil:
                            return DecisionForest.Tree.Path(nodes)

                        nodes.append(nodes[-1][row[nodes[-1].feature_column]])
                        row = row.discard(nodes[-2].feature_column)
                    except IndexError:
                        raise DecisionForest.Tree.Node.ConnectionNotFound(nodes[-1], row)

            def __eq__(self, other: DecisionForest.Tree.Node) -> bool:
                return self._feature_column == other._feature_column
    
            def __hash__(self) -> int:
                return hash(self._feature_column)
    
            connections_begin = '=>('
            connections_end = ')'
            connection_separator = ';'
    
            def __str__(self) -> str:
                return str(self._feature_column)
    
            def __repr__(self) -> str:
                if self.is_leaf():
                    return repr(Dataset.Row.ColumnValue(self._feature_column, self._value))

                return f"{str(self._feature_column)}" \
                       f"{' ' + self.connections_begin}" \
                       f"{f' {self.connection_separator} '.join(repr(connection) for connection in self.connections)}" \
                       f"{self.connections_end}"

            class LeafMissingValueError(SyntaxError):
                def __init__(self, node_string: str):
                    super().__init__(f"Node string ({node_string}) without connections (meaning its a leaf) is "
                                     f"missing a value_or_type.")

            @classmethod
            def parse(cls, string_value: str) -> DecisionForest.Tree.Node:
                """
                >>> DecisionForest.Tree.Node.parse("A: (0, '1') -> 0").is_leaf()
                True

                >>> DecisionForest.Tree.Node.parse("A -> 0").is_leaf()
                True

                >>> repr(DecisionForest.Tree.Node.parse("B: {'b', 'c'} =>(c --> D: {True} =>(True --> S: {0} -> 0))"))
                'B =>(c --> D =>(True --> S -> 0))'
    
                >>> repr(DecisionForest.Tree.Node.parse("B =>(c --> D =>(True --> S: {0} -> 0))"))
                'B =>(c --> D =>(True --> S -> 0))'
    
                :param string_value: string representation of the feature feature_column followed by
                the string representations of the connections_or_value separated by a newline and tab ('\n\t')
                :return: parsed decision tree node
                """
                split = string_value.strip().split(cls.connections_begin, 1)

                feature_string = split[0].strip()
                # If its a feature column value_or_type its a leaf.
                if Dataset.Row.ColumnValue.string_value_separator in feature_string:
                    feature_value = Dataset.Row.ColumnValue.parse(feature_string)
                    return DecisionForest.Tree.Node(feature_value.column, feature_value.value)

                connections = set()
                if len(split) > 1:
                    split[1] = split[1].strip(cls.connections_end)
                    for connection_string in split[1].split(cls.connection_separator):
                        connections.add(DecisionForest.Tree.Connection.parse(connection_string.strip()))
                else:
                    raise DecisionForest.Tree.Node.LeafMissingValueError(string_value)

                if Dataset.Column.string_value_separator in feature_string:
                    feature = Dataset.Column.parse(feature_string)
                else:
                    feature = Dataset.Column(feature_string, [connection.feature_value for connection in connections])
    
                return DecisionForest.Tree.Node(feature, connections)
    
        class Connection(Hashable):
            """
            Represents a :class:`connection<decisionforest.DecisionForest.Tree.Connection>` to
            a :class:`node<decisionforest.DecisionForest.Tree.Node>` in a
            decision :class:`tree<decisionforest.DecisionForest.Tree>`.
            """
            
            def __init__(self, feature_value: Any, node: DecisionForest.Tree.Node):
                self._feature_value = feature_value
                self._node = node
    
            @property
            def feature_value(self) -> Any:
                return self._feature_value
    
            @property
            def node(self) -> DecisionForest.Tree.Node:
                return self._node
    
            def matches(self, feature_value: Any) -> bool:
                return self._feature_value == feature_value
    
            def __eq__(self, other: DecisionForest.Tree.Connection) -> bool:
                return self._feature_value == other._feature_value
    
            def __hash__(self) -> int:
                return hash(self._feature_value)
    
            string_value_separator = '-->'
    
            def __str__(self) -> str:
                return f'{str(self._feature_value)} {self.string_value_separator} {str(self._node)}'
    
            def __repr__(self) -> str:
                return f'{str(self._feature_value)} {self.string_value_separator} {repr(self._node)}'
    
            @classmethod
            def parse(cls, string_value: str) -> DecisionForest.Tree.Connection:
                split = string_value.strip().split(cls.string_value_separator, 1)
                feature_value = safe_literal_eval(split[0].strip())
                node = DecisionForest.Tree.Node.parse(split[1].strip())
                return DecisionForest.Tree.Connection(feature_value, node)
    
        class Path(Sequence["DecisionForest.Tree.Node"]):
            """
            Represents a :class:`path<decisionforest.DecisionForest.Tree.Path>` between multiple
            :class:`nodes<decisionforest.DecisionForest.Tree.Node>` in a
            decision :class:`tree<decisionforest.DecisionForest.Tree>`.
            A path can also be empty or have a single :class:`node<decisionforest.DecisionForest.Tree.Node>`.
            """
            
            class NonSequentialNodeError(ValueError):
                def __init__(self, first: DecisionForest.Tree.Node, second: DecisionForest.Tree.Node):
                    super().__init__(f"Second node ({str(second)}) is not connected to the first node ({repr(first)}).")
    
            def __init__(self, nodes: Optional[Sequence[DecisionForest.Tree.Node]] = None):
                if nodes is not None:
                    if not isinstance(nodes, tuple):
                        nodes = tuple(nodes)
                    self._nodes: Tuple[DecisionForest.Tree.Node, ...] = nodes
                else:
                    self._nodes: Tuple[DecisionForest.Tree.Node, ...] = ()
    
                if len(nodes) > 1:
                    for first, second in zip(self[:-1], self[1:]):
                        if second not in {connection.node for connection in first.connections}:
                            raise DecisionForest.Tree.Path.NonSequentialNodeError(first, second)
    
            @property
            def final_node(self) -> DecisionForest.Tree.Node:
                return self[-1]
    
            @property
            def starting_node(self) -> DecisionForest.Tree.Node:
                return self[0]
    
            @property
            def is_empty(self) -> bool:
                return len(self) == 0
    
            @overload
            def __getitem__(self, index: int) -> DecisionForest.Tree.Node:
                return self._nodes[index]
    
            @overload
            def __getitem__(self, path_slice: slice) -> DecisionForest.Tree.Path:
                return DecisionForest.Tree.Path(self._nodes[path_slice])
    
            def __getitem__(self, index: int) -> DecisionForest.Tree.Node:
                return self._nodes[index]
    
            def __len__(self) -> int:
                return len(self._nodes)
    
            def __eq__(self, other: DecisionForest.Tree.Path) -> bool:
                return self._nodes == other._nodes
    
            node_separator = '==>'
    
            def __str__(self) -> str:
                return f" {self.node_separator} ".join(str(node) for node in self._nodes)
    
            def __repr__(self) -> str:
                return f" {self.node_separator} ".join(repr(node) for node in self._nodes)
    
            @classmethod
            def parse(cls, string_value: str) -> DecisionForest.Tree.Path:
                """
                >>> str(DecisionForest.Tree.Path.parse("B =>(b --> A -> 2) ==> A -> 0"))
                'B ==> A'
    
                :param string_value: node strings separated by a long right arrow ('-->')
                :return: parsed decision tree path
                """
                split = string_value.split(cls.node_separator)
    
                nodes = []
                for node_string in split:
                    nodes.append(DecisionForest.Tree.Node.parse(node_string.strip()))
    
                return DecisionForest.Tree.Path(nodes)
        
        def __init__(self, root: Union[DecisionForest.Tree.Node, DecisionForest.Tree]):
            if isinstance(root, DecisionForest.Tree):
                root = root.root

            self._root: DecisionForest.Tree.Node = root
            
        @property
        def root(self) -> DecisionForest.Tree.Node:
            return self._root

        def iterate(self, row: Dataset.Row) -> DecisionForest.Tree.Path:
            return self._root.iterate(row)

        def __eq__(self, other: DecisionForest.Tree) -> bool:
            return self._root == other._root
        
        def __hash__(self) -> int:
            return hash(self._root)
        
        def __str__(self) -> str:
            return repr(self._root)
        
        def __repr__(self) -> str:
            return repr(self._root)
    
        @classmethod
        def parse(cls, string_value: str) -> DecisionForest.Tree:
            return DecisionForest.Tree(DecisionForest.Tree.Node.parse(string_value))

        class Generator:
            class InadequateDatasetError(ValueError):
                def __init__(self, dataset: Dataset):
                    super().__init__(f"Dataset ({dataset.name}) with "
                                     f"value_or_type frequency ({dataset.column_value_frequency}) is "
                                     f"inadequate for tree generation.")

            @classmethod
            def from_dataset(cls,
                             dataset: Dataset,
                             max_depth: Optional[int] = None) -> DecisionForest.Tree:
                """
                >>> DecisionForest.Tree.Generator.from_dataset(Dataset.from_csv_file(
                ...     Dataset.test_file_paths[Dataset.TestFilePathPicker.Logic],
                ...     Dataset.Column.ValueFrequency.Discrete))
                A =>(False --> F -> False ; True --> C =>(False --> F -> False ; True --> F -> True))

                :param dataset: dataset for which a decision tree will get generated
                :param max_depth: maximum depth of the decision tree
                :return: generated decision tree
                """

                # I think this might work with other frequency types as well.
                if dataset.column_value_frequency == Dataset.Column.ValueFrequency.Discrete:
                    return DecisionForest.Tree(cls._from_discrete_dataset(
                        dataset, dataset, set(dataset.get_feature_columns()), max_depth=max_depth))

                raise DecisionForest.Tree.Generator.InadequateDatasetError(dataset)

            @classmethod
            def _most_frequent_result_column_value(cls, dataset: Dataset) -> Any:
                return max(sorted(dataset.column_value_counts(dataset.result_column).items(),
                                  key=lambda item: item[0]),
                           key=lambda item: item[1])[0]

            @classmethod
            def _most_discriminatory_feature_column(cls,
                                                    dataset: Dataset,
                                                    feature_columns: Iterable[Dataset.Column]) -> Dataset.Column:
                return max_argument(lambda column: information_gain(dataset, column),
                                    sorted(feature_columns, key=lambda feature_column: feature_column.name),
                                    type_limits={Dataset.Column})

            @classmethod
            def _from_discrete_dataset(cls,
                                       parent_dataset: Dataset,
                                       dataset: Dataset,
                                       feature_columns: Set[Dataset.Column],
                                       depth: int = 0,
                                       max_depth: Optional[int] = None) -> DecisionForest.Tree.Node:
                if dataset.is_empty:
                    most_frequent_parent_result_column_value: Any = \
                        cls._most_frequent_result_column_value(parent_dataset)

                    return DecisionForest.Tree.Node(
                        parent_dataset.result_column, most_frequent_parent_result_column_value)

                most_frequent_result_column_value: Any = cls._most_frequent_result_column_value(dataset)

                if len(feature_columns) == 0 or dataset == dataset.where(
                        Dataset.Row.ColumnValue(dataset.result_column, most_frequent_result_column_value)):
                    return DecisionForest.Tree.Node(dataset.result_column, most_frequent_result_column_value)

                most_discriminatory_feature_column: Dataset.Column = \
                    cls._most_discriminatory_feature_column(dataset, feature_columns)

                connections: List[DecisionForest.Tree.Connection] = list()
                for feature_value in most_discriminatory_feature_column:
                    connections.append(DecisionForest.Tree.Connection(
                        feature_value, DecisionForest.Tree.Generator._from_discrete_dataset(
                            dataset,
                            dataset.where(Dataset.Row.ColumnValue(most_discriminatory_feature_column, feature_value)),
                            feature_columns - {most_discriminatory_feature_column},
                            depth=depth + 1,
                            max_depth=max_depth)))

                return DecisionForest.Tree.Node(most_discriminatory_feature_column, connections)

    def __new__(cls, trees: Iterable[DecisionForest.Tree]):
        return super(DecisionForest, cls).__new__(cls, trees)

    # noinspection PyUnusedLocal
    def __init__(self, trees: Iterable[DecisionForest.Tree]):
        super().__init__()
