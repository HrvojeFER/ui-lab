from typing import *
import math


ArgType = TypeVar('ArgType')


def _arg_max(predicate: Callable[[ArgType], Any], arguments: Iterable[ArgType]):
    arguments_iterator = iter(arguments)
    result: str = next(arguments_iterator)
    predicate_result: Any = predicate(result)

    for test in arguments_iterator:
        current_predicate_result = predicate(test)
        if predicate_result is None or current_predicate_result > predicate_result:
            result = test
            predicate_result = current_predicate_result

    return result


def _relative_frequency(dataset: List[List[str]], class_: str):
    return sum(1 for row in dataset if row[-1] == class_) / len(dataset)


def __entropy(dataset: List[List[str]], class_: str):
    probability = _relative_frequency(dataset, class_)
    return probability * math.log2(probability) if probability != 0 else 0


def _entropy(dataset: List[List[str]], classes: Iterable[str]) -> float:
    return -sum(__entropy(dataset, class_) for class_ in classes)


def _information_gain(
        dataset: List[List[str]],
        classes: Iterable[str],
        feature_values: Iterable[str]) -> float:
    return _entropy(dataset, classes) - sum(_entropy(list(row for row in dataset if feature_value in row), classes)
                                            for feature_value in feature_values)


class _Leaf:
    def __init__(self, class_: str):
        self._class: str = class_

    @property
    def class_(self) -> str:
        return self._class

    def __str__(self) -> str:
        return self._class


class _Node:
    def __init__(self, feature: str, children: Iterable[Tuple[str, Union['_Node', _Leaf]]]):
        self._feature: str = feature
        self._children: Set[Tuple[str, Union[_Node, _Leaf]]] = set(children)

    def iterate(self, feature_values: Iterable[str]) -> str:
        feature_values = set(feature_values)
        for child in self._children:
            if child[0] in feature_values:
                if isinstance(child[1], _Node):
                    feature_values = list(feature_values)
                    feature_values.remove(child[0])
                    return child[1].iterate(feature_values)
                else:
                    return child[1].class_

    def __str__(self) -> str:
        return f'{self._feature}:\n\t' + '\n\t'.join(feature_value + ' -> ' + str(child)
                                                     for feature_value, child in self._children)


class ID3:
    # noinspection PyTypeChecker
    def __init__(self, max_depth: int, num_trees: int):
        self._max_depth: int = max_depth
        self._num_trees: int = num_trees
        self._tree: Optional[_Node] = None

    def fit(self, dataset: List[List[str]]) -> NoReturn:
        features = dataset[0][:-1]
        dataset = dataset[1:]
        feature_values = dict((feature, set(row[index] for row in dataset)) for index, feature in enumerate(features))
        classes = set(row[-1] for row in dataset)

        self._tree = self._gen_tree(dataset, dataset, features, feature_values, classes)

    def _gen_tree(self,
                  initial_dataset: List[List[str]],
                  dataset: List[List[str]],
                  features: List[str],
                  feature_values: Mapping[str, Set[str]],
                  classes: Set[str]) -> Union[_Node, _Leaf]:

        if len(dataset) == 0:
            leaf_class = _arg_max(lambda class_: sum(1 for row in initial_dataset if list(row)[-1] == class_), classes)
            return _Leaf(leaf_class)

        leaf_class = _arg_max(lambda class_: sum(1 for row in dataset if row[-1] == class_), classes)
        if len(features) == 0 or sum(1 for row in dataset if row[-1] != leaf_class) == 0:
            return _Leaf(leaf_class)

        feature = _arg_max(lambda f: _information_gain(dataset, classes, feature_values[f]), features)
        subtrees = set()
        for feature_value in feature_values[feature]:
            subtrees.add((feature_value, self._gen_tree(
                initial_dataset,
                [row for row in dataset if feature_value in row],
                [feature for feature in features if feature != feature],
                feature_values,
                classes)))

        return _Node(feature, subtrees)

    def predict(self, dataset: Iterable[Iterable[str]]) -> Generator[str, None, None]:
        for features in dataset:
            yield self._tree.iterate(features)

    def __str__(self) -> str:
        return f'decision_tree_model: name=ID3, max_depth={self._max_depth}, num_trees={self._num_trees}'

    def tree_str(self) -> str:
        return str(self._tree)
