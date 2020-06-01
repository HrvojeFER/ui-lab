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

    def __init__(self, max_depth: int, num_trees: int) -> NoReturn:
        super().__init__()

        self._max_depth: int = max_depth
        self._num_trees: int = num_trees

        self._tree: Optional[DecisionForest.Tree] = None

    def fit(self, dataset: Dataset) -> DecisionForest.Tree:
        super().fit(dataset)

        self._tree = DecisionForest.Tree.Generator.from_dataset(dataset, max_depth=self._max_depth)
        return self._tree

    def print_fitting_results(self, dataset: Dataset) -> NoReturn:
        print(self.fit(dataset))

    def predict(self, dataset: Dataset) -> Generator[DecisionForest.Tree.Path, None, None]:
        super().predict(dataset)

        for row in dataset:
            try:
                yield self._tree.iterate(row)
            except DecisionForest.Tree.Node.ConnectionNotFound:
                yield DecisionForest.Tree.Path()

    def print_prediction_results(self, dataset: Dataset) -> NoReturn:
        print([prediction.final_node.value for prediction in self.predict(dataset)])

    def __str__(self) -> str:
        return f'decision_tree_model: name=ID3, max_depth={self._max_depth}, num_trees={self._num_trees}'

    def tree_str(self) -> str:
        return str(self._tree)


ModelPicker = Enum("ModelPicker", [("ID3", ID3)], type=ModelPicker)
