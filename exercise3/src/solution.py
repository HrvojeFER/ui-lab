import configparser
import csv
import os
import sys

from models import *


data_path = os.path.join('..', 'data')
train_set_path = sys.argv[1]
test_set_path = sys.argv[2]
config_path = sys.argv[3]


config = configparser.ConfigParser()
config_main_section_name = 'main'
with open(os.path.join(data_path, config_path)) as config_file:
    lines = config_file.readlines()
    lines.insert(0, f'[{config_main_section_name}]')
    config.read_file(lines)
    config = dict(config.items(config_main_section_name))

print(config)


if config['model'] == 'ID3':
    model = ID3(int(config['max_depth']), int(config['num_trees']))
else:
    raise RuntimeError(f"No such model: {config['model']}.")

print(model)


train_set = []
with open(os.path.join(data_path, train_set_path)) as dataset_file:
    for row in csv.reader(dataset_file):
        train_set.append(row)

test_set = []
with open(os.path.join(data_path, test_set_path)) as dataset_file:
    for row in csv.reader(dataset_file):
        test_set.append(row)
test_set = test_set[1:]

print(train_set)
print(test_set)


model.fit(train_set)
print(model.tree_str())

predictions = model.predict(test_set)
print([prediction for prediction in predictions])
# TODO: fix this
print(sum(1 for index, prediction in enumerate(predictions) if test_set[index][-1] == prediction) / len(test_set))
