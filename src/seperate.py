from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def seperate_by_label(data):
    sep_data = {}
    for i in range(len(data['image'])):
        example = {'image': data['image'][i].tolist(), 'label': data['label'][i].tolist()}
        lbl = np.argmax(example['label'])
        example['lbl_key'] = lbl.tolist()
        sep_data[lbl] = sep_data.get(lbl, [])
        sep_data[lbl].append(example)
    return sep_data



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_x, train_y, test_x, test_y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

train_data = {'image': train_x, 'label': train_y}
test_data = {'image': test_x, 'label': test_y}

data = seperate_by_label(train_data)

new = {}


for number, info in data.items():
    for i in info:
        new[number] = new.get(number, [])
        new[number].append(0.5*sum(np.array(i['image'])**2))

print(np.mean(new[0]), np.mean(new[1]), np.mean(new[2]), np.mean(new[3]), np.mean(new[4]), np.mean(new[5]), np.mean(new[6]), np.mean(new[7]), np.mean(new[8]), np.mean(new[9]))