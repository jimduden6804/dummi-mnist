from src.constants import *
from unittest import TestCase
from numpy.testing import assert_almost_equal


def distribution_check(distribution):
    sums = []
    for k, probs in distribution.items():
        sums.append(sum(probs))
    return sums

def probs_check(probs):
    return sum([prob for k, prob in probs.items()])

class TestStringMethods(TestCase):

    def test_constants(self):

        assert_almost_equal(distribution_check(ANCHOR_TRANSITION), [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        assert_almost_equal(distribution_check(CONTEXT_LIFT_DISTRIBUTION), [1., 1.])

        self.assertAlmostEqual(probs_check(ANCHOR_PROB), 1.)
        self.assertAlmostEqual(probs_check(CONTEXT_PROB), 1.)