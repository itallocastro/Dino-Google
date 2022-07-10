import numpy as np


def create_one_tiling(feat_range, bins, offset):
    return np.linspace(feat_range[0], feat_range[1], bins + 1)[1:-1] + offset


def get_one_tiling(value, tiling):
    return np.digitize(value, tiling)


def create_tilings(features_range, tilings, bins, offsets):
    total_tilings = []
    for i in range(tilings):
        tiling_current = []
        bins_current = bins[i]
        offsets_current = offsets[i]
        # Para cada dimensão
        for j in range(len(features_range)):
            part_tiling = create_one_tiling(features_range[j], bins_current[j], offsets_current[j])
            tiling_current.append(part_tiling)
        total_tilings.append(tiling_current)
    return np.array(total_tilings)


def discrete(state, bins):
    index = []
    for i in range(len(state)):
        index.append(
            np.digitize(state[i], bins[i])
        )
    return tuple(index)


def get_tile(feature, tilings):
    tile_per_tiling = []
    for tiling in tilings:
        tile_per_tiling.append(discrete(feature, tiling))
    return np.array(tile_per_tiling)


def create_weight(num_tilings, bins, num_actions):
    matrices = []
    for i in range(num_tilings):
        matrices.append(np.zeros(
            tuple(
                np.concatenate((bins[i], [num_actions]))
            )))
    return matrices


def sum_weights(ws, current_state, action, number_tilings):
    total_sum = 0
    for j in range(number_tilings):
        state = tuple(np.concatenate((current_state[j], [action])))
        total_sum = total_sum + ws[j][state]
    return total_sum
