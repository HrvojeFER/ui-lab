from input.file_manager import InputKey
from algorithms.abstract_search import SearchAlgorithmAttributes
from algorithms.helpers import InformedDataManager
from algorithms.factory import SearchAlgorithmFactory, BlindSearchAlgorithmKey, InformedSearchAlgorithmKey
from heuristic_quality_checker import HeuristicQualityChecker
from data_structures.iteration_level import IterationLevel

verbose = False
print('Set verbose to true in the code to view input and search trees.')

ignore = 'ignore'
long = 'minimized'
algorithms = 'algorithms'
check_heuristic = 'check_heuristic'

exercise_run_configuration = \
    {
        InputKey.ai:
            {
                algorithms:
                    [
                        BlindSearchAlgorithmKey.BreadthFirstSearch,
                        BlindSearchAlgorithmKey.UniformCostSearch
                    ],
                check_heuristic: False

            },
        InputKey.ai_fail:
            {
                algorithms:
                    [
                        InformedSearchAlgorithmKey.AStarSearchAlgorithm
                    ],
                check_heuristic: True
            },
        InputKey.ai_pass:
            {
                algorithms:
                    [
                        InformedSearchAlgorithmKey.AStarSearchAlgorithm
                    ],
                check_heuristic: True
            },
        InputKey.istra:
            {
                algorithms:
                    [
                        BlindSearchAlgorithmKey.BreadthFirstSearch,
                        BlindSearchAlgorithmKey.UniformCostSearch
                    ],
                check_heuristic: False
            },
        InputKey.istra_normal:
            {
                algorithms:
                    [
                        InformedSearchAlgorithmKey.AStarSearchAlgorithm
                    ],
                check_heuristic: True
            },
        InputKey.istra_pessimistic:
            {
                algorithms:
                    [
                        InformedSearchAlgorithmKey.AStarSearchAlgorithm
                    ],
                check_heuristic: True
            },
        InputKey.three:
            {
                long: True,
                ignore: False,
                algorithms:
                    [
                        InformedSearchAlgorithmKey.AStarSearchAlgorithm
                    ],
                check_heuristic: True
            }
    }

algorithm_attributes = SearchAlgorithmAttributes(chaotic=False)

for input_key in exercise_run_configuration:
    if ignore in exercise_run_configuration[input_key] and exercise_run_configuration[input_key][ignore]:
        continue

    print('\n-------------------------------------------------------------------------------------------------------\n')

    data_manager = InformedDataManager(input_key)
    state_space = data_manager.state_space
    goal = data_manager.goal
    heuristic = data_manager.heuristic

    print(input_key.value + '\n\n')
    if long not in exercise_run_configuration[input_key]:
        if verbose:
            print('\nState space:\n' + str(state_space))
            print('Goal:\n' + str(goal))
            print('Heuristic:\n' + str(heuristic) + '\n')
    else:
        print('Wait a bit, please. It should take around 5 minutes to fully complete. I hope that is reasonable.\n'
              'The A* algorithm on the 3x3 was really slow so I added open and closed set limits to optimize.\n'
              'The results are quite interesting.\n')

    for algorithm_key in exercise_run_configuration[input_key][algorithms]:
        algorithm = SearchAlgorithmFactory.create_algorithm(algorithm_key, data_manager, algorithm_attributes)
        print(str(algorithm) + '\n')

        treelike_result = algorithm.search()

        if not verbose and long not in exercise_run_configuration[input_key]:
            print('Search tree:\n' + str(treelike_result.stringify(IterationLevel.visited)))

        print('States visited: ' + (str(treelike_result.visit_count) if treelike_result.visit_count is not None else
                                    str(len(treelike_result.visited))) + '\n')
        if treelike_result.path_found:
            print('Path:\n' + str(treelike_result.path) + '\n')
            print('Length: ' + str(len(treelike_result.path)))
            print('Cost: ' + str(treelike_result.path.cost) + '\n\n')
        else:
            print('Path not found!')

    if long in exercise_run_configuration[input_key]:
        print('Wait for the heuristic quality check, please.')

    if exercise_run_configuration[input_key][check_heuristic]:
        heuristic_quality_checker = HeuristicQualityChecker(data_manager)
        quality_check_result = heuristic_quality_checker.check_quality()
        if long in exercise_run_configuration[input_key]:
            print(quality_check_result.stringify_minimized() + '\n')
        else:
            print(str(quality_check_result) + '\n')

    print('\n-------------------------------------------------------------------------------------------------------\n')
