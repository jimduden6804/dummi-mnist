import copy
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_x, train_y, test_x, test_y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

train_data = {'image': train_x, 'label': train_y}
test_data = {'image': test_x, 'label': test_y}



def seperate_by_label(data):
    sep_data = {}
    for i in range(len(data['image'])):
        example = {'image': data['image'][i], 'label': data['label'][i]}
        lbl = np.argmax(example['label'])
        example['lbl_key'] = lbl
        sep_data[lbl] = sep_data.get(lbl, [])
        sep_data[lbl].append(example)
    return sep_data

