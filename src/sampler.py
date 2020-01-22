from random import shuffle

import numpy as np

ANCHOR_DISTRIBUTION = {0: 0.1,
                       1: 0.1,
                       2: 0.1,
                       3: 0.1,
                       4: 0.1,
                       5: 0.1,
                       6: 0.1,
                       7: 0.1,
                       8: 0.1,
                       9: 0.1}

ANCHOR_TRANSITION = {0: [0.35, 0.03, 0.03, 0.03, 0.05, 0.05, 0.11, 0.05, 0.2, 0.1],
                     1: [0.05, 0.3, 0.05, 0.05, 0.15, 0.05, 0.05, 0.2, 0.05, 0.05],
                     2: [0.01, 0.04, 0.4, 0.05, 0.1, 0.02, 0.03, 0.25, 0.05, 0.05],
                     3: [0.015, 0.035, 0.05, 0.35, 0.05, 0.1, 0.05, 0.02, 0.23, 0.1],
                     4: [0.05, 0.07, 0.05, 0.05, 0.29, 0.05, 0.03, 0.18, 0.05, 0.18],
                     5: [0.02, 0.03, 0.05, 0.05, 0.03, 0.3, 0.2, 0.05, 0.07, 0.2],
                     6: [0.07, 0.03, 0.02, 0.03, 0.05, 0.2, 0.25, 0.05, 0.15, 0.15],
                     7: [0.02, 0.2, 0.15, 0.05, 0.1, 0.03, 0.02, 0.35, 0.04, 0.04],
                     8: [0.07, 0.01, 0.03, 0.08, 0.07, 0.1, 0.18, 0.03, 0.25, 0.18],
                     9: [0.05, 0.05, 0.05, 0.1, 0.2, 0.1, 0.05, 0.05, 0.1, 0.25]}

CONTEXT_PROB = {0: 0.5, 1: 0.5}

CONTEXT_LIFT = {0: [0.15, 0.25, 0.07, 0.1, 0.05, 0.03, 0.05, 0.12, 0.08, 0.1],
                1: [0.1, 0.15, 0.12, 0.0, 0.03, 0.05, 0.1, 0.12, 0.25, 0.08]}


def sample_index(prob, rand):
        return np.where(np.less(rand(), np.cumsum(prob)))[0][0]

def sample(distribution, rand):
    prob_vec = [prob for anchor, prob in distribution.items()]
    return sample_index(prob_vec, rand)

def sample_anchor(anchor_distribution, rand):
    return sample(anchor_distribution, rand)

def sample_context(context_prob, rand):
    return sample(context_prob, rand)

def sample_reco(len_list):
    numbers = range(len_list)
    shuffle(numbers)
    return numbers

def create_click_prob(reco, anchor, context, anchor_transition, context_lift):
    probs = []
    for number in reco:
        probs.append(anchor_transition[anchor][number])
    click_prob = np.array(probs) * np.array(context_lift[context]).tolist()
    return click_prob

def sample_click(click_prob, rand):
    return sample_index(click_prob, rand)

rand = lambda: np.random.uniform(0., 1., 1)[0]
anchor = sample_anchor(ANCHOR_DISTRIBUTION, rand)
context = sample(CONTEXT_PROB, rand)
reco = sample_reco(10)
click_prob = create_click_prob(reco anchor, context, ANCHOR_TRANSITION, CONTEXT_LIFT)
click = sample_click(click_prob, rand)






