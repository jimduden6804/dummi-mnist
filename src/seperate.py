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
