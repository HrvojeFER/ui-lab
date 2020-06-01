from __future__ import annotations

from typing import *
from abc import *
from enum import *

from dataset import Dataset
from decisionforest import DecisionForest


class AbstractModel(ABC):
    _name: str = "Abstract Model"

    def __init__(self) -> NoReturn:
        self._is_fit = False

    @abstractmethod
    def fit(self, dataset: Dataset) -> Any:
        self._is_fit = True

    @abstractmethod
    def print_fitting_results(self, dataset: Dataset) -> NoReturn: ...

    class UnfitModelError(RuntimeError):
        def __init__(self, name: str):
            super().__init__(f"Model ({name}) is unfit.")

    @abstractmethod
    def predict(self, dataset: Dataset) -> Any:
        if not self._is_fit:
            raise AbstractModel.UnfitModelError(self._name)

    @abstractmethod
    def print_prediction_results(self, dataset: Dataset) -> NoReturn: ...


class ModelPicker(Enum):
    pass


class ID3(AbstractModel):
    _name = "ID3"

    def __init__(self, **kwargs) -> NoReturn:
        super().__init__()

        try:
            self._max_depth: Optional[int] = kwargs['max_depth']
        except KeyError:
            self._max_depth: Optional[int] = None

        try:
            self._num_trees: int = kwargs['num_trees']
        except KeyError:
            self._num_trees: int = 1

        self._tree: Optional[DecisionForest.Tree] = None

    def fit(self, dataset: Dataset) -> DecisionForest.Tree:
        super().fit(dataset)

        self._tree = DecisionForest.Tree.Generator.from_dataset(dataset, max_depth=self._max_depth)
        return self._tree

    def print_fitting_results(self, dataset: Dataset) -> NoReturn:
        self.fit(dataset)

        def _tree_node_generator(node: DecisionForest.Tree.Node, depth: int = 0) -> \
                Generator[Tuple[DecisionForest.Tree.Node, int], None, None]:
            if node.is_leaf():
                return

            yield node, depth

            for connection in node.connection_tuple:
                yield from _tree_node_generator(connection.node, depth + 1)

        print(', '.join(f"{depth}:{str(node)}" for node, depth in _tree_node_generator(self._tree.root)))

    def predict(self, dataset: Dataset) -> Generator[DecisionForest.Tree.Path, None, None]:
        super().predict(dataset)

        for row in dataset:
            try:
                yield self._tree.iterate(row)
            except DecisionForest.Tree.Node.ConnectionNotFound:
                yield DecisionForest.Tree.Path()

    def print_prediction_results(self, dataset: Dataset) -> NoReturn:
        predictions = list(self.predict(dataset))

        print(' '.join(str(prediction.final_node.value) for prediction in predictions))

        correct_prediction_percentage = sum(1 for prediction_path, row in zip(predictions, dataset)
                                            if prediction_path.final_node.value == row.result_value) / len(dataset)
        print(correct_prediction_percentage)

        for real_row_class_value in dataset.result_column:
            prediction_row = []
            for predicted_column_class_value in dataset.result_column:
                prediction_row.append(sum(
                    1 for real_class_value, prediction in zip(dataset.result_values, predictions)
                    if real_class_value == real_row_class_value and
                    prediction.final_node.value == predicted_column_class_value))

            print(*prediction_row)

    def __str__(self) -> str:
        return f'decision_tree_model: name=ID3, max_depth={self._max_depth}, num_trees={self._num_trees}'

    def tree_str(self) -> str:
        return str(self._tree)


ModelPicker = Enum("ModelPicker", [("ID3", ID3)], type=ModelPicker)
