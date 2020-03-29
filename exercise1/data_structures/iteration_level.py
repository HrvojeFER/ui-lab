from enum import Enum


class IterationLevel(Enum):
    trees = -1,
    roots = 0,
    visited = 1,
    branches = 2,
    leaves = 3
