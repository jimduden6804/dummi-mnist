from src.sampler import *
from unittest import TestCase
from random import sample as rand_sample
from numpy.testing import assert_almost_equal


def gen(lambdas):
    for l in lambdas:
        yield l


sample_fn = lambda num_reco, _: sorted(num_reco, reverse=True)

m_fn = [sample_fn, lambda a, _: [1], lambda a, _: [0, 2]]


class TestStringMethods(TestCase):
    def test_sample_index(self):
        prob1 = [0.2, 0.3, 0.5, 0.]
        prob2 = [0.2, 0.3, 0., 0.5]

        result1 = sample_index(prob1, lambda: 0.51)
        result2 = sample_index(prob2, lambda: 0.6)
        result3 = sample_index(prob2, lambda: 0.0)
        result4 = sample_index(prob1, lambda: 1.0)

        self.assertEqual(result1, 2)
        self.assertEqual(result2, 3)
        self.assertEqual(result3, 0)
        self.assertEqual(result4, 2)

    def test_sample(self):
        anchor_distribution = {0: 0.2,
                               1: 0.3,
                               2: 0.5,
                               3: 0.0}

        result1 = sample(anchor_distribution, lambda: 0.51)

        self.assertEqual(result1, 2)

    def test_sample_reco(self):
        number_recos = 5
        result = sample_reco(number_recos, sample_fn)

        self.assertEqual(result, [4, 3, 2, 1, 0])

    def test_sample_reco_with_random_sample_fn(self):
        number_recos = 3
        result = sample_reco(number_recos, rand_sample)
        expected_set = {0, 1, 2}

        self.assertSetEqual(set(result), expected_set)

    def test_calculate_click_prob(self):
        anchor = 0
        anchor_transition = {0: [0.5, 0.4, 0.1], 1: [], 2: []}

        context = 1
        context_lift = {0: [], 1: [0.5, 0.4, 0.1], 2: []}

        reco = [2, 0, 1]
        expected_clickprob = [0.05, 0.2, 0.04]

        result = calculate_click_prob(reco, anchor, context, anchor_transition, context_lift)

        assert_almost_equal(result, expected_clickprob, 15)

    def test_to_dense(self):
        indices = [1, 5, 9]
        number_recos = 10
        expected_dense_mask = [0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]
        self.assertListEqual(to_dense(indices, number_recos), expected_dense_mask)

    def test_keep_position_mask(self):
        number_recos = 10
        random_lambdas = [lambda a, b: [6], lambda a, b: [0, 5, 6, 7, 9, 8, 1]]
        rand = gen(random_lambdas)
        expected_mask = [1., 1., 0., 0., 0., 1., 1., 1., 1., 1.]
        mask = keep_position_mask(number_recos, sample_fn=lambda a, b: next(rand)(a, b))
        self.assertListEqual(mask, expected_mask)

    def test_generate_example(self):
        lambdas = [lambda: 0.2, lambda: 0.6, lambda: 0.81, lambda: 0.49]
        reco_lengths = [0, 1, 2]
        anchor_prob = {0: 0.4, 1: 0.2, 3: 0.4}
        context_prob = {0: 0.5, 1: 0.5}
        anchor_transition = {0: [0.5, 0.4, 0.1], 1: [], 2: []}
        context_lift = {0: [], 1: [0.5, 0.4, 0.1], 2: []}
        number_recos = 3

        rand = gen(lambdas)
        sample_fns = gen(m_fn)

        result = generate_example(anchor_prob=anchor_prob,
                                  context_prob=context_prob,
                                  anchor_transition=anchor_transition,
                                  context_lift_distribution=context_lift,
                                  number_recos=number_recos,
                                  reco_lengths=reco_lengths,
                                  rand=lambda: next(rand)(),
                                  sample_fn=lambda a, _: next(sample_fns)(a, _))

        expected = {'anchor': 0, 'context': 1, 'click_position': 2, 'reco': [2, 1, 0], 'seen_click_position': 0,
                    'seen_mask': [1., 0., 1.]}
        self.assertDictEqual(result, expected)

    def test_join_example_with_data(self):
        seperated_labels = {0: [{'image': [1., 1., 1.], 'label': [1., 0., 0., 0.], 'lbl_key': 0}],
                            1: [{'image': [2., 2., 2.], 'label': [0., 1., 0., 0.], 'lbl_key': 1}],
                            2: [{'image': [3., 3., 3.], 'label': [0., 0., 1., 0.], 'lbl_key': 2}]}

        example = {'anchor': 0,
                   'context': 1,
                   'click_position': 2,
                   'reco': [2, 1, 0],
                   'seen_click_position': 0,
                   'seen_mask': [1., 0., 1.]}

        expected_new_example = {'anchor': 0,
                                'anchor_image': [1., 1., 1.],
                                'anchor_label': [1., 0., 0., 0.],
                                'anchor_lbl_key': 0,
                                'context': 1,
                                'context_vec': [0., 1.],
                                'click_position': 2,
                                'reco': [2, 1, 0],
                                'seen_click_position': 0,
                                'seen_mask': [1., 0., 1.],
                                'detailed_reco': [{'image': [3., 3., 3.], 'label': [0., 0., 1., 0.], 'lbl_key': 2, 'context': [1., 0., 0.]},
                                                  {'image': [2., 2., 2.], 'label': [0., 1., 0., 0.], 'lbl_key': 1, 'context': [0., 1., 0.]},
                                                  {'image': [1., 1., 1.], 'label': [1., 0., 0., 0.], 'lbl_key': 0, 'context': [0., 0., 1.]}]}



        new_example = join_example_with_data(example, 2, seperated_labels)

        self.assertDictEqual(new_example, expected_new_example)








