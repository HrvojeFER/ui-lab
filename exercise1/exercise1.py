from data import *
from algorithms import *


data_key = DataKeys.three
algorithm_key = AlgorithmKeys.IterativeDepthFirstSearch
algorithm_factory = AlgorithmFactory()


state_space_lines = DataManager.get_state_space_lines(data_key)
state_space = StateSpace.parse(state_space_lines)
print('\nState space:\n' + str(state_space))

heuristic_lines = DataManager.get_heuristics_lines(data_key)
heuristic = Heuristic.parse(heuristic_lines)
print('Heuristic:\n' + str(heuristic))

search_goal_lines = DataManager.get_search_goal_lines(data_key)
search_goal = SearchGoal.parse(search_goal_lines)
print('Search goal:\n' + str(search_goal) + '\n')

search_data_manager = SearchDataManager(state_space, heuristic)

search_algorithm = algorithm_factory.create_algorithm(algorithm_key, search_data_manager)
search_tree = search_algorithm.generate_search_tree(search_goal)
print('Search results:\n')
print('Tree:\n' + str(search_tree))
print('\tState visits: ' + str(len(search_tree.get_visited())) + '\n')

if search_tree.path is not None:
    print('Path:\n' + str(search_tree.path) + '\n')
    print('\tLength: ' + str(len(search_tree.path)))
    print('\tCost: ' + str(search_tree.path.get_cost()))
else:
    print('Path not found!')
