def solution() -> None:
    import configparser
    import os
    import sys

    from dataset import Dataset
    from modelfactory import ModelFactory

    data_path = os.path.join('.')
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

    model = ModelFactory.create(config['model'])

    model.print_fitting_results(Dataset.from_csv_file(train_set_path, Dataset.Column.ValueFrequency.Discrete))
    model.print_prediction_results(Dataset.from_csv_file(test_set_path, Dataset.Column.ValueFrequency.Discrete))


if __name__ == '__main__':
    """
    import sys
    import threading

    sys.setrecursionlimit(100000)
    threading.stack_size(200000000)
    thread = threading.Thread(target=solution())
    thread.start()
    """

    solution()
