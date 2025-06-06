import numpy as np


def traindata_divice(train_data, train_target):
    grouping = int(train_data.shape[0] / 4)
    train_data_before = train_data[:grouping, :]
    train_data_after = train_data[-grouping:, :]
    train_target_before = train_target[:grouping]
    train_target_after = train_target[-grouping:]
    train_data_1 = np.vstack((train_data_before, train_data_after))
    train_target_1 = np.concatenate((train_target_before, train_target_after))
    train_data = train_data[grouping:-grouping, :]
    train_target = train_target[grouping:-grouping]
    return train_data, train_target, train_data_1, train_target_1
