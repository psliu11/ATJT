import random
import torch
import pandas as pd
from gen_neg import gen_neg


def make_weak_labels(batch_size, input, item_count, cate_values, k):
    # hist_item_per = pd.read_csv("movielens_data/hist_item_per.csv", squeeze=True)
    weak_labels_batch = [1] * batch_size
    position = list(range(input.size(0)))
    random.shuffle(position)
    half_length = len(position) // k
    random_half = position[:half_length]
    # if half_length >= 32:
    m = input[:, -1] - random.randint(1, min(input[:, -1]))
    for z in random_half:
        # input[z, (m + 1)[z]] = gen_neg(input, item_count, hist_item_per)
        input[z, (m + 1)[z]] = random.randint(0, item_count - 1)
        input[z, (m + 20)[z]] = cate_values[input[z, (m + 1)[z]]]
        weak_labels_batch[z] = 0
    # elif half_length <= 31:
    # m = input[:, -1] - random.randint(1, min(input[:, -1]))
    # w = input[:, -1] - random.randint(1, min(input[:, -1]))
    # for z in random_half:
    #     input[z, (m + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (m + 20)[z]] = cate_values[input[z, (m + 1)[z]]]
    #     input[z, (w + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (w + 20)[z]] = cate_values[input[z, (w + 1)[z]]]
    #     weak_labels_batch[z] = 0
    # else:
    # m = input[:, -1] - random.randint(1, min(input[:, -1]))
    # w = input[:, -1] - random.randint(1, min(input[:, -1]))
    # q = input[:, -1] - random.randint(1, min(input[:, -1]))
    # for z in random_half:
    #     input[z, (m + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (m + 20)[z]] = cate_values[input[z, (m + 1)[z]]]
    #     input[z, (w + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (w + 20)[z]] = cate_values[input[z, (w + 1)[z]]]
    #     input[z, (q + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (q + 20)[z]] = cate_values[input[z, (q + 1)[z]]]
    #     weak_labels_batch[z] = 0

    # m = input[:, -1] - random.randint(1, min(input[:, -1]))
    # w = input[:, -1] - random.randint(1, min(input[:, -1]))
    # q = input[:, -1] - random.randint(1, min(input[:, -1]))
    # p = input[:, -1] - random.randint(1, min(input[:, -1]))
    # n = input[:, -1] - random.randint(1, min(input[:, -1]))
    # m1 = input[:, -1] - random.randint(1, min(input[:, -1]))
    # w1 = input[:, -1] - random.randint(1, min(input[:, -1]))
    # q1 = input[:, -1] - random.randint(1, min(input[:, -1]))
    # p1 = input[:, -1] - random.randint(1, min(input[:, -1]))
    # n1 = input[:, -1] - random.randint(1, min(input[:, -1]))
    # for z in random_half:
    #     input[z, (m + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (m + 20)[z]] = cate_values[input[z, (m + 1)[z]]]
    #     input[z, (w + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (w + 20)[z]] = cate_values[input[z, (w + 1)[z]]]
    #     input[z, (q + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (q + 20)[z]] = cate_values[input[z, (q + 1)[z]]]
    #     input[z, (p + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (p + 20)[z]] = cate_values[input[z, (p + 1)[z]]]
    #     input[z, (n + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (n + 20)[z]] = cate_values[input[z, (n + 1)[z]]]
    #     input[z, (m1 + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (m1 + 20)[z]] = cate_values[input[z, (m1 + 1)[z]]]
    #     input[z, (w1 + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (w1 + 20)[z]] = cate_values[input[z, (w1 + 1)[z]]]
    #     input[z, (q1 + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (q1 + 20)[z]] = cate_values[input[z, (q1 + 1)[z]]]
    #     input[z, (p1 + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (p1 + 20)[z]] = cate_values[input[z, (p1 + 1)[z]]]
    #     input[z, (n1 + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (n1 + 20)[z]] = cate_values[input[z, (n1 + 1)[z]]]
        weak_labels_batch[z] = 0
    weak_labels_batch = torch.tensor(weak_labels_batch).cuda()

    return input, weak_labels_batch


def make_weak_labels_1(batch_size, input, item_count, cate_values, city_values, postal_values, stars_values, k):
    # hist_item_per = pd.read_csv("movielens_data/hist_item_per.csv", squeeze=True)
    weak_labels_batch = [1] * batch_size
    position = list(range(input.size(0)))
    random.shuffle(position)
    half_length = len(position) // k
    random_half = position[:half_length]
    # if half_length >= 32:
    m = input[:, -1] - random.randint(1, min(input[:, -1]))
    for z in random_half:
        # input[z, (m + 1)[z]] = gen_neg(input, item_count, hist_item_per)
        input[z, (m + 5)[z]] = random.randint(0, item_count - 1)
        input[z, (m + 24)[z]] = cate_values[input[z, (m + 1)[z]]]
        input[z, (m + 43)[z]] = city_values[input[z, (m + 1)[z]]]
        input[z, (m + 62)[z]] = postal_values[input[z, (m + 1)[z]]]
        input[z, (m + 81)[z]] = stars_values[input[z, (m + 1)[z]]]
        weak_labels_batch[z] = 0
    # elif half_length <= 31:
    # m = input[:, -1] - random.randint(1, min(input[:, -1]))
    # w = input[:, -1] - random.randint(1, min(input[:, -1]))
    # for z in random_half:
    #     input[z, (m + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (m + 20)[z]] = cate_values[input[z, (m + 1)[z]]]
    #     input[z, (w + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (w + 20)[z]] = cate_values[input[z, (w + 1)[z]]]
    #     weak_labels_batch[z] = 0
    # else:
    # m = input[:, -1] - random.randint(1, min(input[:, -1]))
    # w = input[:, -1] - random.randint(1, min(input[:, -1]))
    # q = input[:, -1] - random.randint(1, min(input[:, -1]))
    # for z in random_half:
    #     input[z, (m + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (m + 20)[z]] = cate_values[input[z, (m + 1)[z]]]
    #     input[z, (w + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (w + 20)[z]] = cate_values[input[z, (w + 1)[z]]]
    #     input[z, (q + 1)[z]] = random.randint(0, item_count - 1)
    #     input[z, (q + 20)[z]] = cate_values[input[z, (q + 1)[z]]]
    #     weak_labels_batch[z] = 0
    weak_labels_batch = torch.tensor(weak_labels_batch).cuda()

    return input, weak_labels_batch