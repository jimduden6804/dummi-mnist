import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ANCHOR_PROB = {0: 0.1,
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

CONTEXT_PROB = {0: 1., 1: 0.}

CONTEXT_LIFT_DISTRIBUTION = {0: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  #[0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.5, 0.03, 0.02], #[0.15, 0.25, 0.07, 0.1, 0.05, 0.03, 0.05, 0.12, 0.08, 0.1], #[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],   #
                             1: [0.1, 0.15, 0.12, 0.0, 0.03, 0.05, 0.1, 0.12, 0.25, 0.08]}

# transitions
anchor_transition_df = pd.DataFrame(ANCHOR_TRANSITION).transpose()
ax = sns.heatmap(anchor_transition_df, annot=True, cmap="YlGnBu")
ax.set(xlabel='Candidate', ylabel='Anchor', title= 'Click Probabilities')

# position probs
context_probabilities = pd.DataFrame(CONTEXT_LIFT_DISTRIBUTION).drop(columns=1).transpose()
ax = sns.heatmap(context_probabilities, annot=True, cmap="YlGnBu")
ax.set(xlabel='Position', ylabel='context_0', title= 'Position Bias')








