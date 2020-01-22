from tensorflow.python.framework.test_util import TensorFlowTestCase

from src.seperate import seperate_by_label


class TestInput(TensorFlowTestCase):
    def test_seperate_data(self):
        data = {'image': [[1., 1., 1.],
                          [2., 2., 2.],
                          [3., 3., 3.],
                          [1., 1., 1.]],
                'label': [[1., 0., 0., 0.],
                          [0., 1., 0., 0.],
                          [0., 0., 1., 0.],
                          [1., 0., 0., 0.]]}

        seperated_data = {0: [{'image': [1., 1., 1.], 'label': [1., 0., 0., 0.], 'lbl_key': 0},
                              {'image': [1., 1., 1.], 'label': [1., 0., 0., 0.], 'lbl_key': 0}],
                          1: [{'image': [2., 2., 2.], 'label': [0., 1., 0., 0.], 'lbl_key': 1}],
                          2: [{'image': [3., 3., 3.], 'label': [0., 0., 1., 0.], 'lbl_key': 2}]}

        self.assertDictEqual(seperate_by_label(data), seperated_data)
