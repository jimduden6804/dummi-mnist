from random import sample as rand_sample
from src.constants import *
import numpy as np


def sample_index(prob, rand):
    normed_prob = np.cumsum(prob)
    return np.where(np.less_equal(rand(), normed_prob / normed_prob[-1]))[0][0]


def sample(distribution, rand):
    prob_vec = [prob for anchor, prob in distribution.items()]
    return sample_index(prob_vec, rand)


def sample_reco(number_recos, sample_fn=rand_sample):
    labels = list(range(number_recos))
    return sample_fn(labels, number_recos)


def calculate_click_prob(reco, anchor, context, anchor_transition, context_lift):
    probs = []
    for number in reco:
        probs.append(anchor_transition[anchor][number])
    click_prob = np.array(probs) * np.array(context_lift[context])
    return click_prob.tolist()


def generate_example(anchor_prob=ANCHOR_PROB,
                     context_prob=CONTEXT_PROB,
                     anchor_transition=ANCHOR_TRANSITION,
                     context_lift_distribution=CONTEXT_LIFT_DISTRIBUTION,
                     number_recos=10,
                     rand=lambda: np.random.uniform(0., 1., 1)[0],
                     sample_fn=rand_sample):
    anchor = sample(anchor_prob, rand)
    context = sample(context_prob, rand)
    reco = sample_reco(number_recos, sample_fn)
    click_prob = calculate_click_prob(reco, anchor, context, anchor_transition, context_lift_distribution)
    click_position = sample_index(click_prob, rand)
    return {'anchor': anchor, 'context': context, 'reco': reco, 'click_position': click_position}
