from random import sample as rand_sample
from src.constants import *
import numpy as np
import copy


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


def to_dense(indices, number_recos):
    return [1. if index in set(indices) else 0. for index in range(number_recos)]


def keep_position_mask(number_recos, reco_lengths=[6, 7, 8, 9, 10], sample_fn=rand_sample):
    reco_length = sample_fn(reco_lengths, 1)[0]
    positions = list(range(number_recos))
    keep_positions = sample_fn(positions, reco_length)

    return to_dense(keep_positions, number_recos)


def generate_example(anchor_prob=ANCHOR_PROB,
                     context_prob=CONTEXT_PROB,
                     anchor_transition=ANCHOR_TRANSITION,
                     context_lift_distribution=CONTEXT_LIFT_DISTRIBUTION,
                     number_recos=10,
                     reco_lengths=[6, 7, 8, 9, 10],
                     rand=lambda: np.random.uniform(0., 1., 1)[0],
                     sample_fn=rand_sample):
    anchor = sample(anchor_prob, rand)
    context = sample(context_prob, rand)
    reco = sample_reco(number_recos, sample_fn)
    click_prob = calculate_click_prob(reco, anchor, context, anchor_transition, context_lift_distribution)
    click_position = sample_index(click_prob, rand)
    mask = keep_position_mask(number_recos, reco_lengths, sample_fn)
    seen_click_prob = [p * m for p, m in list(zip(click_prob, mask))]
    seen_position = sample_index(seen_click_prob, rand)
    return {'anchor': int(anchor),
            'context': int(context),
            'reco': reco,
            'click_position': int(click_position),
            'seen_click_position': int(seen_position),
            'seen_mask': mask}


def to_context_vec(context, num_contexts):
    contexts = list(range(num_contexts))
    return [1. if ctx in set([context]) else 0. for ctx in contexts]


def add_position_context(detailed_reco):
    for i, reco in enumerate(detailed_reco):
        ctx = len(detailed_reco) * [0.]
        ctx[i] = 1.
        reco['context'] = ctx
    return detailed_reco


def join_example_with_data(example, num_contexts, seperated_labels, sample_fn=rand_sample):
    anchor_data = sample_fn(seperated_labels.get(example['anchor']), 1)[0]
    context_vec = to_context_vec(example['context'], num_contexts)
    detailed_reco = [sample_fn(seperated_labels.get(ex), 1)[0] for ex in example['reco']]
    detailed_reco = add_position_context(copy.deepcopy(detailed_reco))
    example.update({
        'context_vec': context_vec,
        'anchor_image': anchor_data['image'],
        'anchor_lbl_key': anchor_data['lbl_key'],
        'anchor_label': anchor_data['label'],
        'detailed_reco': detailed_reco})
    return example

