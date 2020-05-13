from tensorflow.examples.tutorials.mnist import input_data
import gzip
import json
import os
import shutil
from src.sampler import generate_example, join_example_with_data
from src.seperate import seperate_by_label
import logging



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_x, train_y, test_x, test_y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

train_data = {'image': train_x, 'label': train_y}
test_data = {'image': test_x, 'label': test_y}


job_name = 'dummi-mnist'
version = 1
version_str = str(version).zfill(5)
cwd = '/'.join(os.getcwd().split('/')[:-1])
data_dir = f'{cwd}/data'
path = f'{data_dir}/{job_name}-{version_str}'


number_train_examples = 10000
number_test_examples = 2500

def create_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else: os.makedirs(path)


def write_file(file_type, data, number_examples):
    seperated_data = seperate_by_label(data)

    with gzip.open(f'{path}/{file_type}.gz', 'w') as file:
        for i in range(number_examples):
            if i % 1000 == 0: print(f'{i} examples have been written')
            example = generate_example()
            example_with_data = join_example_with_data(example, 2, seperated_data)
            file.write((json.dumps(example_with_data)+'\n').encode())

def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    create_directory(path)

    logging.info(f'writing train data to {path}')
    write_file('train', train_data, number_train_examples)

    logging.info(f'writing test data to {path}')
    write_file('test', test_data, number_test_examples)



if __name__ == '__main__':
    main()

