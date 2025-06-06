import numpy as np


# 生成负样本
def gen_neg(pos_list, item_count, hist_item_per):
    neg = pos_list[0]
    while neg in pos_list:
        neg = np.random.choice(item_count, p=hist_item_per)
    return neg
