from __future__ import annotations

from typing import *

from dataset import Dataset
from extensions import safe_literal_eval


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
                         connections: Union[Iterable[DecisionForest.Tree.Connection],
                                            Mapping[Any, DecisionForest.Tree.Node]]):
                self._feature_column: Dataset.Column = feature_column
    
                if isinstance(connections, Mapping):
                    self._connections: Dict[Any, DecisionForest.Tree.Node] = dict(connections)
                else:
                    connections = dict((connection.feature_value, connection.node) for connection in connections)
                    self._connections: Dict[Any, DecisionForest.Tree.Node] = connections
    
                for connection in self.connections:
                    if not feature_column.supports_value_or_type(connection.feature_value):
                        raise Dataset.Column.UnsupportedValueError(feature_column, connection.feature_value)
    
            @property
            def feature_column(self) -> Dataset.Column:
                return self._feature_column
    
            @property
            def connections(self) -> FrozenSet[DecisionForest.Tree.Connection]:
                return frozenset(DecisionForest.Tree.Connection(feature_value, node)
                                 for feature_value, node in self._connections.items())
    
            def is_leaf(self) -> bool:
                return len(self) == 0
    
            def __getitem__(self, feature_value: Any) -> DecisionForest.Tree.Node:
                return self._connections[feature_value]
    
            def __len__(self) -> int:
                return len(self._connections)
    
            def __iter__(self) -> Iterator[DecisionForest.Tree.Connection]:
                return iter(self.connections)
    
            class UnknownFeatureValueError(ValueError):
                def __init__(self, node: DecisionForest.Tree.Node, feature_value: Any):
                    super().__init__(f"Feature value ({str(feature_value)}) is not present in the "
                                     f"nodes ({str(node)}) connections ({str(node.connections)}).")
    
            def iterate(self, feature_values: Sequence[Any]) -> DecisionForest.Tree.Path:
                """
                >>> node = DecisionForest.Tree.Node.parse("B =>(c -> D =>(True -> S); d -> A)")
                >>> str(node.iterate(['d']))
                'B --> A'

                >>> str(node.iterate(['c']))
                'B --> D'

                >>> str(node.iterate(['c', True]))
                'B --> D --> S'

                :param feature_values: feature values with which to iterate over the nodes
                :return: path the nodes took
                """

                nodes = [self]
    
                for feature_value in feature_values:
                    try:
                        nodes.append(nodes[-1][feature_value])
                    except KeyError:
                        raise DecisionForest.Tree.Node.UnknownFeatureValueError(nodes[-1], feature_value)
    
                return DecisionForest.Tree.Path(nodes)
    
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
                return f"{str(self._feature_column)}" \
                       f"{' ' + self.connections_begin if not self.is_leaf() else ''}" \
                       f"{f' {self.connection_separator} '.join(repr(connection) for connection in self.connections)}" \
                       f"{self.connections_end if not self.is_leaf() else ''}"
    
            @classmethod
            def parse(cls, string_value: str) -> DecisionForest.Tree.Node:
                """
                >>> DecisionForest.Tree.Node.parse("A: (0, '1')").is_leaf()
                True

                >>> repr(DecisionForest.Tree.Node.parse("B: {'b', 'c'}=>(c -> D: {True}=>(True -> S: {None}))"))
                'B =>(c -> D =>(True -> S))'
    
                >>> repr(DecisionForest.Tree.Node.parse("B =>(c -> D =>(True -> S))"))
                'B =>(c -> D =>(True -> S))'
    
                :param string_value: string representation of the feature column followed by
                the string representations of the connections separated by a newline and tab ('\n\t')
                :return: parsed decision tree node
                """
                split = string_value.strip().split(cls.connections_begin, 1)
    
                feature_column_string = split[0].strip()
                if ':' in feature_column_string:
                    feature_column = Dataset.Column.parse(feature_column_string)
                else:
                    feature_column = None
    
                connections = set()
                if len(split) > 1:
                    split[1] = split[1].strip(cls.connections_end)
                    for connection_string in split[1].split(cls.connection_separator):
                        connections.add(DecisionForest.Tree.Connection.parse(connection_string.strip()))
    
                if feature_column is None:
                    feature_column = Dataset.Column(feature_column_string,
                                                    (connection.feature_value for connection in connections))
    
                return DecisionForest.Tree.Node(feature_column, connections)
    
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
    
            string_value_separator = '->'
    
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
    
            def __init__(self, nodes: Sequence[DecisionForest.Tree.Node]):
                if not isinstance(nodes, list):
                    nodes = list(nodes)
                self._nodes: List[nodes] = nodes
    
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
    
            node_separator = '-->'
    
            def __str__(self) -> str:
                return f" {self.node_separator} ".join(str(node) for node in self._nodes)
    
            def __repr__(self) -> str:
                return f"\n{self.node_separator}\n".join(repr(node) for node in self._nodes)
    
            @classmethod
            def parse(cls, string_value: str) -> DecisionForest.Tree.Path:
                """
                >>> str(DecisionForest.Tree.Path.parse("B =>(b -> A)-->A"))
                'B --> A'
    
                :param string_value: node strings separated by a long right arrow ('-->')
                :return: parsed decision tree path
                """
                split = string_value.split(cls.node_separator)
    
                nodes = []
                for node_string in split:
                    nodes.append(DecisionForest.Tree.Node.parse(node_string.strip()))
    
                return DecisionForest.Tree.Path(nodes)
        
        def __init__(self, root: DecisionForest.Tree.Node):
            self._root = root
            
        @property
        def root(self) -> DecisionForest.Tree.Node:
            return self._root
        
        def __eq__(self, other: DecisionForest.Tree) -> bool:
            return self._root == other._root
        
        def __hash__(self) -> int:
            return hash(self._root)
        
        def __str__(self) -> str:
            return str(self._root)
        
        def __repr__(self) -> str:
            return repr(self._root)
    
        @classmethod
        def parse(cls, string_value: str) -> DecisionForest.Tree:
            return DecisionForest.Tree(DecisionForest.Tree.Node.parse(string_value))

    def __new__(cls, trees: Iterable[DecisionForest.Tree]):
        return super(DecisionForest, cls).__new__(cls, trees)

    # noinspection PyUnusedLocal
    def __init__(self, trees: Iterable[DecisionForest.Tree]):
        super().__init__()
