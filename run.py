import random

import os
import json
import pandas as pd
import numpy as np
import torch.utils.data as Data
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, RandomSampler
import torch
import torch.optim as optim
from tqdm import tqdm

import metrics
from build_map import build_map
from compare.din import DIN, DIN_1
from Make_Weak_labels import make_weak_labels, make_weak_labels_1
from gen_neg import gen_neg
from metrics import Metric
from cross_entropy import con_model_cross_entropy, base_model_cross_entropy, denoise_model_cross_entropy, l2_model_cross_entropy
from Traindata_Divide import traindata_divice

from findweight import FindWeight, FindWeight_1

from compare.DCN import DCN, DCN_1
from compare.wide_deep import Wide_Deep, Wide_Deep_1
from compare.FEARec import FEARec, FEARec_1
from compare.SASRec import SASRec
from compare.S3Rec import S3RecModel
from compare.S3Rec_PretrainDataset import PretrainDataset
from pad_sequences import pad_sequences
from utils import to_df


# # 读取reviews_df
# reviews_df = to_df('D:\\amazon\\Electronics_5.json')
#
# # 对reviews_df的用户以及购买时间进行排序并选取前100000个用户购买行为
# reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
# # reviews_df = reviews_df.iloc[:100000, :]
#
# # 读取meta_df并选出前100000个用户购买行为包含的商品
# meta_df = to_df('D:\\amazon\\meta_Electronics.json')
# meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
#
# # 选取reviews_df和meta_df需要的标签
# reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
# meta_df = meta_df[['asin', 'categories']]
# meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])
#
# # 获取asin、categories、reviewerID的个数和字典并用字典进行排序映射
# asin_map, asin_key = build_map(meta_df, 'asin')
# cate_map, cate_key = build_map(meta_df, 'categories')
# revi_map, revi_key = build_map(reviews_df, 'reviewerID')
#
# # 获取长度
# user_count, item_count, cate_count, example_count = \
#     len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
# np.save("amazon_data/amazon_user_count", user_count)
# np.save("amazon_data/amazon_item_count", item_count)
# np.save("amazon_data/amazon_cate_count", cate_count)
#
# # 用asin_map对reviews_df的asin映射
# reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
#
# # meta_df的asin排序以及索引排序
# meta_df = meta_df.sort_values('asin')
# meta_df = meta_df.reset_index(drop=True)
#
# # reviews_df的索引排序
# reviews_df = reviews_df.reset_index(drop=True)
#
# # 将meta_df的asin与categories进行映射
# d1 = meta_df.to_dict()
# meta_dict = d1['categories']
# np.save("amazon_data/cate_values", np.array(list(meta_dict.values())))
#
# # 每个商品出现的概率，用于负采样
# hist_item_num = reviews_df['asin'].value_counts()
# hist_item_per = hist_item_num / hist_item_num.sum()
# hist_item_per = hist_item_per ** 0.75
# hist_item_per = hist_item_per / hist_item_per.sum()
# hist_item_per.to_csv("amazon_data/hist_item_per.csv", index=False)
#
# neg_data_candidate = []
# for i in range(153922):
#     neg_data_n = []
#     # 生产负候选样本
#     for j in range(19):
#         neg_item = np.random.choice(item_count, p=hist_item_per)
#         neg_data_n.append(neg_item)
#     neg_data_candidate.append(neg_data_n)
# np.save("amazon_data/amazon_neg_data_candidate", neg_data_candidate)


# # 读取reviews_df
# data_file = open("D:\\yelp\\yelp_academic_dataset_review.json", encoding='UTF-8')
# data = []
# for line in data_file:
#     data.append(json.loads(line))
# reviews_df = pd.DataFrame(data)
# data_file.close()
#
# # n可设置为整个训练集行为的大小，当前为100个行为
# reviews_df = reviews_df.sort_values(['user_id', 'date'])
# reviews_df = reviews_df.iloc[:10000, :]
#
# # 读取meta_df并选出前n个用户购买行为包含的商品
# data_file = open("D:\\yelp\\yelp_academic_dataset_business.json", encoding='UTF-8')
# data = []
# for line in data_file:
#     data.append(json.loads(line))
# meta_df = pd.DataFrame(data)
# data_file.close()
# meta_df = meta_df[meta_df['business_id'].isin(reviews_df['business_id'].unique())]
#
# # 读取meta_df并选出前n个用户购买行为包含的商品
# data_file = open("D:\\yelp\\yelp_academic_dataset_user.json", encoding='UTF-8')
# data = []
# for line in data_file:
#     data.append(json.loads(line))
# user_df = pd.DataFrame(data)
# data_file.close()
# user_df = user_df[user_df['user_id'].isin(reviews_df['user_id'].unique())]
#
# # 选取reviews_df和meta_df需要的标签
# reviews_df = reviews_df[['user_id', 'business_id', 'date']]
# meta_df = meta_df[['business_id', 'city', 'postal_code', 'stars', 'categories']]
# user_df = user_df[['user_id', 'useful', 'funny', 'cool', 'average_stars']]
# meta_df['categories'] = meta_df['categories'].str.split(',').str[0]
# meta_df['categories'] = meta_df['categories'].fillna('0')
#
# # 获取asin、categories、reviewerID的个数和字典并用字典进行排序映射
# asin_map, asin_key = build_map(meta_df, 'business_id')
# city_map, city_key = build_map(meta_df, 'city')
# postal_map, postal_key = build_map(meta_df, 'postal_code')
# stars_map, stars_key = build_map(meta_df, 'stars')
# cate_map, cate_key = build_map(meta_df, 'categories')
# revi_map, revi_key = build_map(user_df, 'user_id')
# useful_map, useful_key = build_map(user_df, 'useful')
# funny_map, funny_key = build_map(user_df, 'funny')
# cool_map, cool_key = build_map(user_df, 'cool')
# average_stars_map, average_stars_key = build_map(user_df, 'average_stars')
#
# # 获取长度
# user_count, item_count, cate_count, city_count, postal_count, stars_count, useful_count, funny_count, cool_count, \
#     average_stars_count, example_count = len(revi_map), len(asin_map), len(cate_map), len(city_map), len(postal_map), \
#     len(stars_map), len(useful_map), len(funny_map), len(cool_map), len(average_stars_map), reviews_df.shape[0]
# np.save("yelp_data/yelp_user_count", user_count)
# np.save("yelp_data/yelp_item_count", item_count)
# np.save("yelp_data/yelp_cate_count", cate_count)
# np.save("yelp_data/yelp_city_count", city_count)
# np.save("yelp_data/yelp_postal_count", postal_count)
# np.save("yelp_data/yelp_stars_count", stars_count)
# np.save("yelp_data/yelp_useful_count", useful_count)
# np.save("yelp_data/yelp_funny_count", funny_count)
# np.save("yelp_data/yelp_cool_count", cool_count)
# np.save("yelp_data/yelp_average_stars_count", average_stars_count)
#
# # 用asin_map对reviews_df的asin映射
# reviews_df['business_id'] = reviews_df['business_id'].map(lambda x: asin_map[x])
# reviews_df['user_id'] = reviews_df['user_id'].map(revi_map)
# reviews_df = reviews_df.dropna(subset=['user_id'])
#
# # meta_df的asin排序以及索引排序
# meta_df = meta_df.sort_values('business_id')
# meta_df = meta_df.reset_index(drop=True)
#
# # reviews_df的索引排序
# reviews_df = reviews_df.reset_index(drop=True)
#
# # meta_df的asin排序以及索引排序
# user_df = user_df.sort_values('user_id')
# user_df = user_df.reset_index(drop=True)
#
# # 将meta_df进行映射
# d1 = meta_df.to_dict()
# city_dict = d1['city']
# np.save("yelp_data/city_values", np.array(list(city_dict.values())))
# postal_dict = d1['postal_code']
# np.save("yelp_data/postal_values", np.array(list(postal_dict.values())))
# stars_dict = d1['stars']
# np.save("yelp_data/stars_values", np.array(list(stars_dict.values())))
# meta_dict = d1['categories']
# np.save("yelp_data/cate_values", np.array(list(meta_dict.values())))
#
# # 将user_df进行映射
# d2 = user_df.to_dict()
# useful_dict = d2['useful']
# funny_dict = d2['funny']
# cool_dict = d2['cool']
# average_stars_dict = d2['average_stars']
#
# # 每个商品出现的概率，用于负采样
# hist_item_num = reviews_df['business_id'].value_counts()
# hist_item_per = hist_item_num / hist_item_num.sum()
# hist_item_per = hist_item_per ** 0.75
# hist_item_per = hist_item_per / hist_item_per.sum()
# hist_item_per.to_csv("yelp_data/hist_item_per.csv", index=False)
#
# pos_data, neg_data_candidate = [], []
#
# # 设置max_sl
# max_sl = 19
# np.save("yelp_data/yelp_max_sl", max_sl)
# for reviewerID, hist in reviews_df.groupby('user_id'):
#     pos_list = hist['business_id'].tolist()
#
#     if len(pos_list) >= 5:
#         for i in range(1, len(pos_list)):
#             # 生成每一次的历史记录，即之前的浏览历史
#             hist = pos_list[:i]
#             hist_gender, hist_city, hist_postal, hist_stars = [], [], [], []
#
#             if i == len(pos_list) - 1:
#                 if len(hist) <= 19:
#                     sl = len(hist)
#                     for j in hist:
#                         hist_gender.append(meta_dict[j])
#                         hist_city.append(city_dict[j])
#                         hist_postal.append(postal_dict[j])
#                         hist_stars.append(stars_dict[j])
#                     hist = pad_sequences(hist, maxlen=max_sl)
#                     hist_gender = pad_sequences(hist_gender, maxlen=max_sl)
#                     hist_city = pad_sequences(hist_city, maxlen=max_sl)
#                     hist_postal = pad_sequences(hist_postal, maxlen=max_sl)
#                     hist_stars = pad_sequences(hist_stars, maxlen=max_sl)
#                     pos_data.append([reviewerID, useful_dict[reviewerID], funny_dict[reviewerID], cool_dict[reviewerID],
#                                     average_stars_dict[reviewerID]] + hist + hist_gender + hist_city + hist_postal + hist_stars + [pos_list[i],
#                                     meta_dict[pos_list[i]], city_dict[pos_list[i]], postal_dict[pos_list[i]],
#                                     stars_dict[pos_list[i]], sl])
#                 else:
#                     hist = hist[-19:]
#                     sl = len(hist)
#                     for j in hist:
#                         hist_gender.append(meta_dict[j])
#                         hist_city.append(city_dict[j])
#                         hist_postal.append(postal_dict[j])
#                         hist_stars.append(stars_dict[j])
#                     pos_data.append([reviewerID, useful_dict[reviewerID], funny_dict[reviewerID], cool_dict[reviewerID],
#                                     average_stars_dict[reviewerID]] + hist + hist_gender + hist_city + hist_postal + hist_stars + [pos_list[i],
#                                     meta_dict[pos_list[i]], city_dict[pos_list[i]], postal_dict[pos_list[i]],
#                                     stars_dict[pos_list[i]], sl])
#         neg_data_n = []
#         # 生产负候选样本
#         for i in range(63):
#             neg_item = gen_neg(pos_list, item_count, hist_item_per)
#             neg_data = [neg_item, meta_dict[neg_item], city_dict[neg_item], postal_dict[neg_item], stars_dict[neg_item]]
#             neg_data_n.append(neg_data)
#         neg_data_candidate.append(neg_data_n)
#
# train_pos_data, test_pos_data = [], []
# train_neg_data_candidate, test_neg_data_candidate = [], []
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# # 索引
# for train_index, test_index in kf.split(pos_data):
#     train_pos_data.append(np.array(pos_data)[train_index])
#     train_neg_data_candidate.append(np.array(neg_data_candidate)[train_index])
#     test_pos_data.append(np.array(pos_data)[test_index])
#     test_neg_data_candidate.append(np.array(neg_data_candidate)[test_index])
#
# # 不同的数量n这里需要修改
# train_pos_data[3] = np.delete(train_pos_data[3], -1, 0)
# train_pos_data[4] = np.delete(train_pos_data[4], -1, 0)
# train_pos_data = np.array(train_pos_data)
# train_neg_data_candidate[3] = np.delete(train_neg_data_candidate[3], -1, 0)
# train_neg_data_candidate[4] = np.delete(train_neg_data_candidate[4], -1, 0)
# train_neg_data_candidate = np.array(train_neg_data_candidate)
# test_pos_data[0] = np.delete(test_pos_data[0], -1, 0)
# test_pos_data[1] = np.delete(test_pos_data[1], -1, 0)
# test_pos_data[2] = np.delete(test_pos_data[2], -1, 0)
# test_pos_data = np.array(test_pos_data)
# test_neg_data_candidate[0] = np.delete(test_neg_data_candidate[0], -1, 0)
# test_neg_data_candidate[1] = np.delete(test_neg_data_candidate[1], -1, 0)
# test_neg_data_candidate[2] = np.delete(test_neg_data_candidate[2], -1, 0)
# test_neg_data_candidate = np.array(test_neg_data_candidate)
#
# train_data, train_target = [], []
# for i in range(5):
#     train_data_ = train_pos_data[i].copy()
#     train_target_ = [1 for m in range(len(train_pos_data[i]))]
#     for j in range(1):
#         train_pos_data[i][:, -6:-1] = train_neg_data_candidate[i][:, j]
#         train_data_ = np.append(train_data_, train_pos_data[i], axis=0)
#         train_target_ += [0 for m in range(len(train_pos_data[i]))]
#     train_data.append(train_data_)
#     train_target.append(train_target_)
#
# np.save("yelp_data/yelp_train_data", train_data)
# np.save("yelp_data/yelp_train_target", train_target)
#
# np.save("yelp_data/yelp_test_data/test_pos_data", test_pos_data)
# np.save("yelp_data/yelp_test_data/test_neg_data_candidate", test_neg_data_candidate)
#
# test_pos_data = np.load("yelp_data/yelp_test_data/test_pos_data.npy")[2]
# test_neg_data_candidate = np.load("yelp_data/yelp_test_data/test_neg_data_candidate.npy")[2]
# test_pos_data = np.expand_dims(test_pos_data, axis=1)
# test_data_ = test_pos_data.copy()
# test_data_ = test_data_.repeat(63, axis=1)
# test_data_[:, :, -6:-1] = test_neg_data_candidate
# test_data = np.append(test_pos_data, test_data_, axis=1)
# test_data = np.reshape(test_data, (-1, 106))
# np.save("yelp_data/yelp_test_data/yelp_test_data_split2", test_data)
#
# test_target = [1]
# for m in range(63):
#     test_target.append(0)
# c = test_target.copy()
# for i in range(len(test_neg_data_candidate)-1):
#     test_target.extend(c)
# np.save("yelp_data/yelp_test_data/yelp_test_target", test_target)


# reviews_df = pd.read_csv('D:\\movielens\\ratings.csv')
# reviews_df = reviews_df.sort_values(['userId', 'timestamp'])
#
# meta_df = pd.read_csv('D:\\movielens\\movies.csv')
# meta_df = meta_df[meta_df['movieId'].isin(reviews_df['movieId'].unique())]
#
# reviews_df = reviews_df[['userId', 'movieId', 'timestamp']]
# meta_df = meta_df[['movieId', 'genres']]
# meta_df['genres'] = meta_df['genres'].map(lambda x: x.split('|')[0])
#
# asin_map, asin_key = build_map(meta_df, 'movieId')
# cate_map, cate_key = build_map(meta_df, 'genres')
# revi_map, revi_key = build_map(reviews_df, 'userId')
#
# user_count, item_count, cate_count, example_count = \
#     len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
# np.save("movielens_data/movielens_user_count", user_count)
# np.save("movielens_data/movielens_item_count", item_count)
# np.save("movielens_data/movielens_cate_count", cate_count)
#
# reviews_df['movieId'] = reviews_df['movieId'].map(lambda x: asin_map[x])
#
# meta_df = meta_df.reset_index(drop=True)
# reviews_df = reviews_df.reset_index(drop=True)
#
# d1 = meta_df.to_dict()
# meta_dict = d1['genres']
# np.save("movielens_data/cate_values", np.array(list(meta_dict.values())))
#
# hist_item_num = reviews_df['movieId'].value_counts()
# hist_item_per = hist_item_num / hist_item_num.sum()
# hist_item_per = hist_item_per ** 0.75
# hist_item_per = hist_item_per / hist_item_per.sum()
# hist_item_per.to_csv("movielens_data/hist_item_per.csv", index=False)
#
# pos_data, neg_data_candidate = [], []
#
# max_sl = 50
# for reviewerID, hist in reviews_df.groupby('userId'):
#     pos_list = hist['movieId'].tolist()
#     max_sl = min(max_sl, len(pos_list))
# np.save("movielens_data/movielens_max_sl", max_sl-1)
#
# for reviewerID, hist in reviews_df.groupby('userId'):
#     pos_list = hist['movieId'].tolist()
#
#     for i in range(1, len(pos_list)):
#         # 生成每一次的历史记录，即之前的浏览历史
#         if i == len(pos_list) - 1:
#             hist = pos_list[i-19:i]
#             sl = len(hist)
#             hist_gender = []
#             for j in hist:
#                 hist_gender.append(meta_dict[j])
#             hist = pad_sequences(hist, maxlen=max_sl-1)
#             hist_gender = pad_sequences(hist_gender, maxlen=max_sl-1)
#             pos_data.append([reviewerID] + hist + hist_gender + [pos_list[i], meta_dict[pos_list[i]], sl])
#
#     neg_data_n = []
#     # 生产负候选样本
#     for i in range(1):
#         neg_item = gen_neg(pos_list, item_count, hist_item_per)
#         neg_data = [neg_item, meta_dict[neg_item]]
#         neg_data_n.append(neg_data)
#     neg_data_candidate.append(neg_data_n)
#
# train_pos_data, test_pos_data = [], []
# train_neg_data_candidate, test_neg_data_candidate = [], []
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# # 索引
# for train_index, test_index in kf.split(pos_data):
#     train_pos_data.append(np.array(pos_data)[train_index])
#     train_neg_data_candidate.append(np.array(neg_data_candidate)[train_index])
#     test_pos_data.append(np.array(pos_data)[test_index])
#     test_neg_data_candidate.append(np.array(neg_data_candidate)[test_index])
#
# train_pos_data[3] = np.delete(train_pos_data[3], -1, 0)
# train_pos_data[4] = np.delete(train_pos_data[4], -1, 0)
# train_pos_data = np.array(train_pos_data)
# train_neg_data_candidate[3] = np.delete(train_neg_data_candidate[3], -1, 0)
# train_neg_data_candidate[4] = np.delete(train_neg_data_candidate[4], -1, 0)
# train_neg_data_candidate = np.array(train_neg_data_candidate)
# test_pos_data[0] = np.delete(test_pos_data[0], -1, 0)
# test_pos_data[1] = np.delete(test_pos_data[1], -1, 0)
# test_pos_data[2] = np.delete(test_pos_data[2], -1, 0)
# test_pos_data = np.array(test_pos_data)
# test_neg_data_candidate[0] = np.delete(test_neg_data_candidate[0], -1, 0)
# test_neg_data_candidate[1] = np.delete(test_neg_data_candidate[1], -1, 0)
# test_neg_data_candidate[2] = np.delete(test_neg_data_candidate[2], -1, 0)
# test_neg_data_candidate = np.array(test_neg_data_candidate)
#
# np.save("movielens_data/movielens_test_data/movielens_test_pos_data", test_pos_data)
# np.save("movielens_data/movielens_test_data/movielens_test_neg_data_candidate", test_neg_data_candidate)
#
# test_pos_data = np.load("movielens_data/movielens_test_data/movielens_test_pos_data.npy")
# test_neg_data_candidate = np.load("movielens_data/movielens_test_data/movielens_test_neg_data_candidate.npy")
# test_pos_data = test_pos_data[3]
# test_neg_data_candidate = test_neg_data_candidate[3]
# test_pos_data = np.expand_dims(test_pos_data, axis=1)
# test_data_ = test_pos_data.copy()
# test_data_ = test_data_.repeat(63, axis=1)
# test_data_[:, :, -3:-1] = test_neg_data_candidate
# test_data = np.append(test_pos_data, test_data_, axis=1)
# test_data = np.reshape(test_data, (-1, 42))
# np.save("movielens_data/movielens_test_data/movielens_test_data_split3", test_data)
#
# test_target = [1]
# for m in range(63):
#     test_target.append(0)
# c = test_target.copy()
# for i in range(27697):
#     test_target.extend(c)
# np.save("movielens_data/movielens_test_data/movielens_test_target", test_target)
#
# train_data, train_target = [], []
# for i in range(5):
#     train_data_ = train_pos_data[i].copy()
#     train_target_ = [1 for m in range(len(train_pos_data[i]))]
#     for j in range(1):
#         train_pos_data[i][:, -3:-1] = train_neg_data_candidate[i][:, j]
#         train_data_ = np.append(train_data_, train_pos_data[i], axis=0)
#         train_target_ += [0 for m in range(len(train_pos_data[i]))]
#     train_data.append(train_data_)
#     train_target.append(train_target_)
#
# np.save("movielens_data/movielens_train_data", train_data)
# np.save("movielens_data/movielens_train_target", train_target)


# # 读取reviews_df
# reviews_df = to_df('D:\\amazon\\Electronics_5.json')
#
# # 对reviews_df的用户以及购买时间进行排序并选取前100000个用户购买行为
# reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
# # reviews_df = reviews_df.iloc[:100000, :]
#
# # 读取meta_df并选出前100000个用户购买行为包含的商品
# meta_df = to_df('D:\\amazon\\meta_Electronics.json')
# meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
#
# # 选取reviews_df和meta_df需要的标签
# reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
# meta_df = meta_df[['asin', 'categories']]
# meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])
#
# # 获取asin、categories、reviewerID的个数和字典并用字典进行排序映射
# asin_map, asin_key = build_map(meta_df, 'asin')
# cate_map, cate_key = build_map(meta_df, 'categories')
# revi_map, revi_key = build_map(reviews_df, 'reviewerID')
#
# # 获取长度
# user_count, item_count, cate_count, example_count = \
#     len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
# np.save("amazon_data/amazon_user_count", user_count)
# np.save("amazon_data/amazon_item_count", item_count)
# np.save("amazon_data/amazon_cate_count", cate_count)
#
# # 用asin_map对reviews_df的asin映射
# reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
#
# # meta_df的asin排序以及索引排序
# meta_df = meta_df.sort_values('asin')
# meta_df = meta_df.reset_index(drop=True)
#
# # reviews_df的索引排序
# reviews_df = reviews_df.reset_index(drop=True)
#
# # 将meta_df的asin与categories进行映射
# d1 = meta_df.to_dict()
# meta_dict = d1['categories']
# np.save("amazon_data/cate_values", np.array(list(meta_dict.values())))
#
# # 每个商品出现的概率，用于负采样
# hist_item_num = reviews_df['asin'].value_counts()
# hist_item_per = hist_item_num / hist_item_num.sum()
# hist_item_per = hist_item_per ** 0.75
# hist_item_per = hist_item_per / hist_item_per.sum()
# hist_item_per.to_csv("amazon_data/hist_item_per.csv", index=False)
#
# pos_data, neg_data_candidate = [], []
#
# # 设置max_sl
# max_sl = 19
# np.save("amazon_data/amazon_max_sl", max_sl)
# for reviewerID, hist in reviews_df.groupby('reviewerID'):
#     pos_list = hist['asin'].tolist()
#
#     for i in range(1, len(pos_list)):
#         # 生成每一次的历史记录，即之前的浏览历史
#         hist = pos_list[:i]
#         hist_gender = []
#
#         if i == len(pos_list) - 1:
#             if len(hist) <= 19:
#                 sl = len(hist)
#                 for j in hist:
#                     hist_gender.append(meta_dict[j])
#                 hist = pad_sequences(hist, maxlen=max_sl)
#                 hist_gender = pad_sequences(hist_gender, maxlen=max_sl)
#                 pos_data.append([reviewerID] + hist + hist_gender + [pos_list[i], meta_dict[pos_list[i]], sl])
#             else:
#                 hist = hist[-19:]
#                 sl = len(hist)
#                 for j in hist:
#                     hist_gender.append(meta_dict[j])
#                 pos_data.append([reviewerID] + hist + hist_gender + [pos_list[i], meta_dict[pos_list[i]], sl])
#
#     neg_data_n = []
#     # 生产负候选样本
#     for i in range(63):
#         neg_item = gen_neg(pos_list, item_count, hist_item_per)
#         neg_data = [neg_item, meta_dict[neg_item]]
#         neg_data_n.append(neg_data)
#     neg_data_candidate.append(neg_data_n)
#
# train_pos_data, test_pos_data = [], []
# train_neg_data_candidate, test_neg_data_candidate = [], []
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# # 索引
# for train_index, test_index in kf.split(pos_data):
#     train_pos_data.append(np.array(pos_data)[train_index])
#     train_neg_data_candidate.append(np.array(neg_data_candidate)[train_index])
#     test_pos_data.append(np.array(pos_data)[test_index])
#     test_neg_data_candidate.append(np.array(neg_data_candidate)[test_index])
#
# train_pos_data[3] = np.delete(train_pos_data[3], -1, 0)
# train_pos_data[4] = np.delete(train_pos_data[4], -1, 0)
# train_pos_data = np.array(train_pos_data)
# train_neg_data_candidate[3] = np.delete(train_neg_data_candidate[3], -1, 0)
# train_neg_data_candidate[4] = np.delete(train_neg_data_candidate[4], -1, 0)
# train_neg_data_candidate = np.array(train_neg_data_candidate)
# test_pos_data[0] = np.delete(test_pos_data[0], -1, 0)
# test_pos_data[1] = np.delete(test_pos_data[1], -1, 0)
# test_pos_data[2] = np.delete(test_pos_data[2], -1, 0)
# test_pos_data = np.array(test_pos_data)
# test_neg_data_candidate[0] = np.delete(test_neg_data_candidate[0], -1, 0)
# test_neg_data_candidate[1] = np.delete(test_neg_data_candidate[1], -1, 0)
# test_neg_data_candidate[2] = np.delete(test_neg_data_candidate[2], -1, 0)
# test_neg_data_candidate = np.array(test_neg_data_candidate)
#
# train_data, train_target = [], []
# for i in range(5):
#     train_data_ = train_pos_data[i].copy()
#     train_target_ = [1 for m in range(len(train_pos_data[i]))]
#     for j in range(1):
#         train_pos_data[i][:, -3:-1] = train_neg_data_candidate[i][:, j]
#         train_data_ = np.append(train_data_, train_pos_data[i], axis=0)
#         train_target_ += [0 for m in range(len(train_pos_data[i]))]
#     train_data.append(train_data_)
#     train_target.append(train_target_)
#
# np.save("amazon_data/amazon_train_data", train_data)
# np.save("amazon_data/amazon_train_target", train_target)
#
# np.save("amazon_data/amazon_test_data/test_pos_data", test_pos_data)
# np.save("amazon_data/amazon_test_data/test_neg_data_candidate", test_neg_data_candidate)
#
# test_pos_data = np.load("amazon_data/amazon_test_data/test_pos_data.npy")[2]
# test_neg_data_candidate = np.load("amazon_data/amazon_test_data/test_neg_data_candidate.npy")[2]
# test_pos_data = np.expand_dims(test_pos_data, axis=1)
# test_data_ = test_pos_data.copy()
# test_data_ = test_data_.repeat(63, axis=1)
# test_data_[:, :, -3:-1] = test_neg_data_candidate
# test_data = np.append(test_pos_data, test_data_, axis=1)
# test_data = np.reshape(test_data, (-1, 42))
# np.save("amazon_data/amazon_test_data/amazon_test_data_split2", test_data)
#
# test_target = [1]
# for m in range(63):
#     test_target.append(0)
# c = test_target.copy()
# for i in range(38479):
#     test_target.extend(c)
# np.save("amazon_data/amazon_test_data/amazon_test_target", test_target)

if __name__ == "__main__":
    def run(dataset, split_n, test, update, batch_size, test_batch_size, shuffle, epochs_1, epochs_2, log_name, con):
        for i in [split_n]:
            if dataset == "amazon":
                train_data = np.load("amazon_data/amazon_train_data.npy")[i]
                train_target = np.load("amazon_data/amazon_train_target.npy")[i]
                test_data = np.load(test)
                test_target = np.load("amazon_data/amazon_test_data/amazon_test_target.npy")
                user_count = np.load("amazon_data/amazon_user_count.npy")
                item_count = np.load("amazon_data/amazon_item_count.npy")
                cate_count = np.load("amazon_data/amazon_cate_count.npy")
                max_sl = np.load("amazon_data/amazon_max_sl.npy")
                cate_values = np.load("amazon_data/cate_values.npy")
            elif dataset == "movielens":
                train_data = np.load("movielens_data/movielens_train_data.npy")[i]
                train_target = np.load("movielens_data/movielens_train_target.npy")[i]
                test_data = np.load(test)
                test_target = np.load("movielens_data/movielens_test_data/movielens_test_target.npy")
                user_count = np.load("movielens_data/movielens_user_count.npy")
                item_count = np.load("movielens_data/movielens_item_count.npy")
                cate_count = np.load("movielens_data/movielens_cate_count.npy")
                max_sl = np.load("movielens_data/movielens_max_sl.npy")
                cate_values = np.load("movielens_data/cate_values.npy")
            else:
                train_data = np.load("yelp_data/yelp_train_data.npy")[i]
                train_target = np.load("yelp_data/yelp_train_target.npy")[i]
                test_data = np.load(test)
                test_target = np.load("yelp_data/yelp_test_data/yelp_test_target.npy")
                user_count = np.load("yelp_data/yelp_user_count.npy")
                item_count = np.load("yelp_data/yelp_item_count.npy")
                cate_count = np.load("yelp_data/yelp_cate_count.npy")
                city_count = np.load("yelp_data/yelp_city_count.npy")
                postal_count = np.load("yelp_data/yelp_postal_count.npy")
                stars_count = np.load("yelp_data/yelp_stars_count.npy")
                useful_count = np.load("yelp_data/yelp_useful_count.npy")
                cool_count = np.load("yelp_data/yelp_cool_count.npy")
                funny_count = np.load("yelp_data/yelp_funny_count.npy")
                average_stars_count = np.load("yelp_data/yelp_average_stars_count.npy")
                max_sl = np.load("yelp_data/yelp_max_sl.npy")
                cate_values = np.load("yelp_data/cate_values.npy")
                city_values = np.load("yelp_data/city_values.npy")
                postal_values = np.load("yelp_data/postal_values.npy")
                stars_values = np.load("yelp_data/stars_values.npy")

            # SASRec
            # train_data_ = train_data[:int(len(train_data)/2), 5:24]
            # train_data_ = train_data[:114934, 1:20]
            # train_data_l = np.array([np.concatenate((arr[arr == 0], arr[arr != 0])) for arr in train_data_])
            # train_data_l = np.array([np.concatenate((arr, arr2)) for arr, arr2 in zip(train_data_l, train_data[:, 100].reshape(-1, 1))])
            # train_data_l = np.array([np.concatenate((arr, arr2)) for arr, arr2 in zip(train_data_l, train_data[:, 39].reshape(-1, 1))])
            # train_data_t = train_data_l[:, :19]
            # pos_data = train_data_l[:, 1:]
            # neg_data = np.load("amazon_data/amazon_neg_data_candidate.npy")[:int(len(train_data)/2), :]
            # mask = np.array([np.where(array > 1, 1, array) for array in train_data_t])
            # pos_data = mask * pos_data
            # neg_data = mask * neg_data
            # train_tensor_data_t = Data.TensorDataset(torch.from_numpy(train_data_t[:train_data_t.shape[0] // 2]).type(torch.LongTensor).cuda())
            # trainloader_t = DataLoader(train_tensor_data_t, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            # train_tensor_data_t_1 = Data.TensorDataset(torch.from_numpy(train_data_t[train_data_t.shape[0] // 2:]).type(torch.LongTensor).cuda())
            # trainloader_t_1 = DataLoader(train_tensor_data_t_1, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            # pos_tensor_data = Data.TensorDataset(torch.from_numpy(pos_data[:pos_data.shape[0] // 2]).type(torch.LongTensor).cuda())
            # posloader = DataLoader(pos_tensor_data, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            # pos_tensor_data_1 = Data.TensorDataset(torch.from_numpy(pos_data[pos_data.shape[0] // 2:]).type(torch.LongTensor).cuda())
            # posloader_1 = DataLoader(pos_tensor_data_1, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            # neg_tensor_data = Data.TensorDataset(torch.from_numpy(neg_data[:neg_data.shape[0] // 2]).type(torch.LongTensor).cuda())
            # negloader = DataLoader(neg_tensor_data, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            # neg_tensor_data_1 = Data.TensorDataset(torch.from_numpy(neg_data[neg_data.shape[0] // 2:]).type(torch.LongTensor).cuda())
            # negloader_1 = DataLoader(neg_tensor_data_1, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            # .....

            train_data, train_target, train_data_1, train_target_1 = traindata_divice(train_data, train_target)  # 添加的训练集划分

            train_tensor_data = Data.TensorDataset(torch.from_numpy(train_data_1[:train_data_1.shape[0] // 2]).type(torch.LongTensor).cuda())
            trainloader = DataLoader(train_tensor_data, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            train_tensor_data_1 = Data.TensorDataset(torch.from_numpy(train_data[:train_data.shape[0] // 2]).type(torch.LongTensor).cuda())
            trainloader_1 = DataLoader(train_tensor_data_1, batch_size=batch_size, shuffle=shuffle, num_workers=0)

            train_tensor_data_neg = Data.TensorDataset(torch.from_numpy(train_data[train_data.shape[0] // 2:]).type(torch.LongTensor).cuda())
            trainloader_neg = DataLoader(train_tensor_data_neg, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            train_tensor_data_neg_1 = Data.TensorDataset(torch.from_numpy(train_data_1[train_data_1.shape[0] // 2:]).type(torch.LongTensor).cuda())
            trainloader_neg_1 = DataLoader(train_tensor_data_neg_1, batch_size=batch_size, shuffle=shuffle, num_workers=0)

            test_tensor_data = Data.TensorDataset(torch.from_numpy(test_data).type(torch.LongTensor).cuda(),
                                                  torch.from_numpy(test_target).type(torch.LongTensor).cuda())
            testloader = DataLoader(test_tensor_data, batch_size=test_batch_size, shuffle=False, num_workers=0)


            # long_sequence_1 = (train_data_t[train_data_t.shape[0] // 2:]).reshape((train_data_t[train_data_t.shape[0] // 2:]).size)
            # pretrain_dataset_1 = PretrainDataset(item_count, max_sl, list(train_data_t[train_data_t.shape[0] // 2:]), list(long_sequence_1))
            # pretrain_dataloader_1 = DataLoader(pretrain_dataset_1, batch_size=256)
            # long_sequence = (train_data_t[:train_data_t.shape[0] // 2]).reshape((train_data_t[:train_data_t.shape[0] // 2]).size)
            # pretrain_dataset = PretrainDataset(item_count, max_sl, list(train_data_t[:train_data_t.shape[0] // 2]), list(long_sequence))
            # pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=256)


            torch.manual_seed(seed=42)
            torch.cuda.manual_seed_all(seed=42)
            random.seed(42)

            train_steps_per_epoch = len(trainloader)
            train_steps_per_epoch_1 = len(trainloader_1)
            test_steps_per_epoch = len(testloader)

            print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
                len(train_tensor_data), len(test_target), train_steps_per_epoch))

            # model = Wide_Deep(train_data.shape[1]-1, max_sl, user_count, item_count, cate_count)
            # model = Wide_Deep_1(train_data.shape[1]-1, max_sl, user_count, item_count, cate_count, city_count,
            #                     postal_count, stars_count, useful_count, cool_count, funny_count, average_stars_count)
            # model = DCN(train_data.shape[1]-1, max_sl, user_count, item_count, cate_count)
            # model = DCN_1(train_data.shape[1]-1, max_sl, user_count, item_count, cate_count, city_count, postal_count,
            #             stars_count, useful_count, cool_count, funny_count, average_stars_count)
            model = DIN(user_count, item_count, cate_count, max_sl)
            # model = DIN_1(user_count, item_count, cate_count, max_sl, city_count, postal_count, stars_count,
            #               useful_count, cool_count, funny_count, average_stars_count)
            # model = FEARec(user_count, item_count, cate_count, max_sl)
            # model = FEARec_1(user_count, item_count, cate_count, max_sl, city_count, postal_count, stars_count,
            #                  useful_count, cool_count, funny_count, average_stars_count)
            # model = SASRec(item_count, max_sl)
            # model = S3RecModel(item_count, max_sl)
            # model.load_state_dict(torch.load(os.path.join('output/', f'{"S3Rec"}-{"movielens"}' + '.pt')))
            model = model.cuda()

            # 添加的噪声识别模型初始化
            denoise_model = FindWeight(user_count, item_count, cate_count, max_sl)
            # denoise_model = FindWeight_1(user_count, item_count, cate_count, max_sl, city_count, postal_count,
            #                              stars_count, useful_count, cool_count, funny_count, average_stars_count)
            denoise_model = denoise_model.cuda()

            denoise_model_D = FindWeight(user_count, item_count, cate_count, max_sl)
            # denoise_model_D = FindWeight_1(user_count, item_count, cate_count, max_sl, city_count, postal_count,
            #                              stars_count, useful_count, cool_count, funny_count, average_stars_count)
            denoise_model_D = denoise_model_D.cuda()

            optimizer_model = optim.Adagrad(model.parameters(), lr=0.000815, weight_decay=0.0005)
            # scheduler = optim.lr_scheduler.StepLR(optimizer_model, step_size=600, gamma=0.7)

            optimizer_denoise_model = optim.Adagrad(denoise_model.parameters(), lr=0.023, weight_decay=0.0005)
            optimizer_denoise_model_D = optim.Adagrad(denoise_model_D.parameters(), lr=0.025, weight_decay=0.0005)

            train_test_log = []  # 存储训练结果
            if update:
                # 用trainloader_1的数据训练推荐模型
                for epoch in range(epochs_1):
                    train_epoch_denoise_loss = 0
                    train_epoch_base_auc, train_epoch_denoise_auc = 0, 0
                    k_sum_p, k_ave_p, k_epoch_ave_p, k_epoch_ave_mean_p, k = 0, 0, 0, 0, 0
                    k_sum_n, k_ave_n, k_epoch_ave_n, k_epoch_ave_mean_n = 0, 0, 0, 0
                    k_2_p, k_2_n, k_epoch_var_p, k_epoch_var_n, k_epoch_var_mean_p, k_epoch_var_mean_n = 0, 0, 0, 0, 0, 0
                    # SASRec
                    # for input, input_1, input_t_1, input_pos_1, input_neg_1 in zip(tqdm(trainloader), trainloader_1, trainloader_t_1, posloader_1, negloader_1):
                    #     input_t_1 = input_t_1[0]
                    #     input_pos_1 = input_pos_1[0]
                    # .....
                    # other
                    for input, input_1, input_neg_1 in zip(tqdm(trainloader), trainloader_1, trainloader_neg_1):
                    # ...
                        input = input[0]
                        input_1 = input_1[0]
                        input_neg_1 = input_neg_1[0]
                        batch_size = input.size(0)
                        optimizer_model.zero_grad()
                        optimizer_denoise_model.zero_grad()

                        input_denoise, weak_labels = make_weak_labels(batch_size, input, item_count, cate_values, 2)
                        # input_denoise, weak_labels = make_weak_labels_1(batch_size, input, item_count, cate_values,
                        #                                                 city_values, postal_values, stars_values, 2)
                        click_batch_weight = denoise_model(input_denoise)

                        click_batch_weight_1 = denoise_model(input_1)

                        # w均值
                        k += 1
                        k_sum_p = (click_batch_weight * weak_labels).sum()
                        k_ave_p = k_sum_p / (batch_size / 2)
                        k_epoch_ave_p += k_ave_p
                        k_epoch_ave_mean_p = k_epoch_ave_p / k
                        k_sum_n = (click_batch_weight * (- (weak_labels - 1))).sum()
                        k_ave_n = k_sum_n / (batch_size / 2)
                        k_epoch_ave_n += k_ave_n
                        k_epoch_ave_mean_n = k_epoch_ave_n / k
                        # w方差
                        non_zero_values_p = (click_batch_weight * weak_labels)[click_batch_weight * weak_labels != 0]
                        k_2_p = torch.var(non_zero_values_p.float() - k_ave_p)
                        k_epoch_var_p += k_2_p
                        k_epoch_var_mean_p = k_epoch_var_p / k
                        non_zero_values_n = (click_batch_weight * (- (weak_labels - 1)))[click_batch_weight * (- (weak_labels - 1)) != 0]
                        k_2_n = torch.var(non_zero_values_n.float() - k_ave_n)
                        k_epoch_var_n += k_2_n
                        k_epoch_var_mean_n = k_epoch_var_n / k

                        # other
                        input_base_1 = torch.cat([input_1, input_neg_1], dim=0)
                        click_batch_weight_1 = torch.cat([click_batch_weight_1, click_batch_weight_1], dim=0)
                        y_pred = model(input_base_1)
                        # ...
                        target_base_1 = torch.tensor(np.concatenate((np.ones(batch_size), np.zeros(batch_size)))).cuda()

                        # SASRec
                        # click_batch_weight_1 = torch.cat((click_batch_weight_1, torch.tensor([0.5] * click_batch_weight_1.shape[0]).cuda()), dim=0)
                        # y_pred = model(input_t_1, input_pos_1, input_neg_1)
                        # .....
                        if epoch < 0:
                            loss_denoise_model = denoise_model_cross_entropy(regular=click_batch_weight,
                                                                             labels=weak_labels)
                        else:
                            # 和推荐模型一起训练去噪模型，多目标为了找更适合权重
                            if con:
                                loss_denoise_model = con_model_cross_entropy(y_pred, target_base_1,
                                                                             weight=click_batch_weight_1,
                                                                             regular=click_batch_weight,
                                                                             labels=weak_labels)
                            # 单独训练推荐模型
                            else:
                                loss_denoise_model = base_model_cross_entropy(y_pred, target_base_1,
                                                                              weight=click_batch_weight_1.detach())
                                # 给每个序列加一个没有辅助任务的权重，第一个epochs就训练
                                # loss_denoise_model = l2_model_cross_entropy(y_pred, target_base_1, weight=click_batch_weight_1)
                        # SASRec
                        # pred_list = y_pred.tolist()
                        # .....
                        # other
                        pred_list = y_pred[:, 1].tolist()
                        # ...
                        target_list = target_base_1.tolist()
                        evaluate_base = Metric(pred_list, target_list)
                        train_epoch_base_auc += evaluate_base.auc()

                        train_epoch_denoise_loss += loss_denoise_model.tolist()

                        weight_list = click_batch_weight.tolist()
                        labels_list = weak_labels.tolist()
                        evaluate_denoise = Metric(weight_list, labels_list)
                        train_epoch_denoise_auc += evaluate_denoise.auc()

                        loss_denoise_model.backward()

                        if epoch < 0:
                            optimizer_denoise_model.step()
                        else:
                            optimizer_model.step()
                            # scheduler.step()
                            optimizer_denoise_model.step()

                    print(f"k_epoch_ave_mean_p {k_epoch_ave_mean_p}, k_epoch_ave_mean_n {k_epoch_ave_mean_n}, "
                          f"k_epoch_var_mean_p {k_epoch_var_mean_p}, k_epoch_var_mean_n {k_epoch_var_mean_n}")
                    train_epoch_denoise_loss /= train_steps_per_epoch_1
                    train_epoch_denoise_auc /= train_steps_per_epoch_1
                    train_epoch_base_auc /= train_steps_per_epoch_1
                    print(f"Epoch {epoch}, train denoise loss {train_epoch_denoise_loss}, "
                          f"train denoise auc {train_epoch_denoise_auc}, train base auc {train_epoch_base_auc}")
                    train_test_log.append(f"Epoch {epoch}, train denoise loss {train_epoch_denoise_loss}, "
                                          f"train denoise auc {train_epoch_denoise_auc}, train base auc {train_epoch_base_auc}")

                # 用trainloader的数据训练推荐模型
                for epoch in range(epochs_2):
                    train_epoch_denoise_loss = 0
                    train_epoch_base_auc, train_epoch_denoise_auc = 0, 0
                    k_sum_p, k_ave_p, k_epoch_ave_p, k_epoch_ave_mean_p, k = 0, 0, 0, 0, 0
                    k_sum_n, k_ave_n, k_epoch_ave_n, k_epoch_ave_mean_n = 0, 0, 0, 0
                    k_2_p, k_2_n, k_epoch_var_p, k_epoch_var_n, k_epoch_var_mean_p, k_epoch_var_mean_n = 0, 0, 0, 0, 0, 0
                    # SASRec
                    # for input, input_1, input_t, input_pos, input_neg in zip(tqdm(trainloader), trainloader_1, trainloader_t, posloader, negloader):
                    #     input_t = input_t[0]
                    #     input_pos = input_pos[0]
                    # .....
                    # other
                    for input, input_1, input_neg in zip(tqdm(trainloader), trainloader_1, trainloader_neg):
                    # ...
                        input = input[0]
                        input_1 = input_1[0]
                        input_neg = input_neg[0]
                        batch_size = input.size(0)
                        optimizer_model.zero_grad()
                        optimizer_denoise_model_D.zero_grad()

                        input_denoise_1, weak_labels_1 = make_weak_labels(batch_size, input_1, item_count, cate_values,
                                                                          2)
                        # input_denoise_1, weak_labels_1 = make_weak_labels_1(batch_size, input, item_count, cate_values,
                        #                                                 city_values, postal_values, stars_values, 2)
                        click_batch_weight_1 = denoise_model_D(input_denoise_1)

                        # w均值
                        k += 1
                        k_sum_p = (click_batch_weight_1 * weak_labels_1).sum()
                        k_ave_p = k_sum_p / (batch_size / 2)
                        k_epoch_ave_p += k_ave_p
                        k_epoch_ave_mean_p = k_epoch_ave_p / k
                        k_sum_n = (click_batch_weight_1 * (- (weak_labels_1 - 1))).sum()
                        k_ave_n = k_sum_n / (batch_size / 2)
                        k_epoch_ave_n += k_ave_n
                        k_epoch_ave_mean_n = k_epoch_ave_n / k
                        # w方差
                        non_zero_values_p = (click_batch_weight_1 * weak_labels_1)[click_batch_weight_1 * weak_labels_1 != 0]
                        k_2_p = torch.var(non_zero_values_p.float() - k_ave_p)
                        k_epoch_var_p += k_2_p
                        k_epoch_var_mean_p = k_epoch_var_p / k
                        non_zero_values_n = (click_batch_weight_1 * (- (weak_labels_1 - 1)))[click_batch_weight_1 * (- (weak_labels_1 - 1)) != 0]
                        k_2_n = torch.var(non_zero_values_n.float() - k_ave_n)
                        k_epoch_var_n += k_2_n
                        k_epoch_var_mean_n = k_epoch_var_n / k

                        click_batch_weight = denoise_model_D(input)
                        # other
                        input_base = torch.cat([input, input_neg], dim=0)
                        click_batch_weight = torch.cat([click_batch_weight, click_batch_weight], dim=0)
                        y_pred = model(input_base)  # 基础推荐模型
                        # ...
                        target_base = torch.tensor(np.concatenate((np.ones(batch_size), np.zeros(batch_size)))).cuda()
                        # SASRec
                        # click_batch_weight = torch.cat((click_batch_weight, torch.tensor([0.5] * click_batch_weight.shape[0]).cuda()), dim=0)
                        # y_pred = model(input_t, input_pos, input_neg)
                        # .....
                        if epoch < 0:
                            loss_denoise_model = denoise_model_cross_entropy(regular=click_batch_weight_1,
                                                                             labels=weak_labels_1)
                        else:
                            # 和推荐模型一起训练去噪模型，多目标为了找更适合权重
                            if con:
                                loss_denoise_model = con_model_cross_entropy(y_pred, target_base,
                                                                             weight=click_batch_weight,
                                                                             regular=click_batch_weight_1,
                                                                             labels=weak_labels_1)
                            # 单独训练推荐模型
                            else:
                                loss_denoise_model = base_model_cross_entropy(y_pred, target_base,
                                                                              weight=click_batch_weight.detach())
                                # 给每个序列加一个没有辅助任务的权重，第一个epochs就训练
                                # loss_denoise_model = l2_model_cross_entropy(y_pred, target_base, weight=click_batch_weight)
                        # SASRec
                        # pred_list = y_pred.tolist()
                        # .....
                        # other
                        pred_list = y_pred[:, 1].tolist()
                        # ...
                        target_list = target_base.tolist()
                        evaluate_base = Metric(pred_list, target_list)
                        train_epoch_base_auc += evaluate_base.auc()

                        train_epoch_denoise_loss += loss_denoise_model.tolist()

                        weight_list = click_batch_weight_1.tolist()
                        labels_list = weak_labels_1.tolist()
                        evaluate_denoise = Metric(weight_list, labels_list)
                        train_epoch_denoise_auc += evaluate_denoise.auc()

                        loss_denoise_model.backward()

                        if epoch < 0:
                            optimizer_denoise_model_D.step()
                        else:
                            optimizer_model.step()
                            # scheduler.step()
                            optimizer_denoise_model_D.step()

                    print(f"k_epoch_ave_mean_p {k_epoch_ave_mean_p}, k_epoch_ave_mean_n {k_epoch_ave_mean_n}, "
                          f"k_epoch_var_mean_p {k_epoch_var_mean_p}, k_epoch_var_mean_n {k_epoch_var_mean_n}")
                    train_epoch_denoise_loss /= train_steps_per_epoch_1
                    train_epoch_denoise_auc /= train_steps_per_epoch_1
                    train_epoch_base_auc /= train_steps_per_epoch_1
                    print(f"Epoch {epoch}, train denoise loss {train_epoch_denoise_loss}, "
                          f"train denoise auc {train_epoch_denoise_auc}, train base auc {train_epoch_base_auc}")
                    train_test_log.append(f"Epoch {epoch}, train denoise loss {train_epoch_denoise_loss}, "
                                          f"train denoise auc {train_epoch_denoise_auc}, train base auc {train_epoch_base_auc}")

            # else:
            #     for epoch in range(epochs_1):
            #         train_epoch_base_loss = 0
            #         for i, batch in tqdm(enumerate(pretrain_dataloader_1), total=len(pretrain_dataloader_1)):
            #             batch = tuple(t.to("cuda") for t in batch)
            #             masked_item_sequence, pos_items, neg_items, \
            #                 masked_segment_sequence, pos_segment, neg_segment = batch
            #             aap_loss, mip_loss, map_loss, sp_loss = model.pretrain(masked_item_sequence, pos_items,
            #                                                                    neg_items, masked_segment_sequence,
            #                                                                    pos_segment, neg_segment)
            #             loss_base_model = 0.2 * aap_loss + 1 * mip_loss + 1 * map_loss + 0.5 * sp_loss
            #             train_epoch_base_loss += loss_base_model.tolist()
            #             loss_base_model.backward()
            #
            #             optimizer_model.step()
            #             # scheduler.step()
            #
            #         train_epoch_base_loss /= train_steps_per_epoch_1
            #
            #     for epoch in range(epochs_2):
            #         train_epoch_base_loss = 0
            #         for i, batch in tqdm(enumerate(pretrain_dataloader), total=len(pretrain_dataloader)):
            #             batch = tuple(t.to("cuda") for t in batch)
            #             masked_item_sequence, pos_items, neg_items, \
            #                 masked_segment_sequence, pos_segment, neg_segment = batch
            #             aap_loss, mip_loss, map_loss, sp_loss = model.pretrain(masked_item_sequence, pos_items,
            #                                                                    neg_items, masked_segment_sequence,
            #                                                                    pos_segment,neg_segment)
            #             loss_base_model = 0.2 * aap_loss + 1 * mip_loss + 1 * map_loss + 0.5 * sp_loss
            #             train_epoch_base_loss += loss_base_model.tolist()
            #             loss_base_model.backward()
            #
            #             optimizer_model.step()
            #             # scheduler.step()
            #
            #         train_epoch_base_loss /= train_steps_per_epoch_1
            #
            #     torch.save(model.state_dict(), os.path.join('output/', f'{"S3Rec"}-{"movielens"}' + '.pt'))

            else:
                for epoch in range(epochs_1):
                    train_epoch_base_loss = 0
                    train_epoch_base_auc = 0
                    # SASRec
                    # for input_t_1, input_pos_1, input_neg_1 in zip(tqdm(trainloader_t_1), posloader_1, negloader_1):
                    #     input_t_1 = input_t_1[0]
                    #     input_pos_1 = input_pos_1[0]
                    # .....
                    # other
                    for input_1, input_neg_1 in zip(tqdm(trainloader_1), trainloader_neg_1):
                        input_1 = input_1[0]
                    # ...
                        input_neg_1 = input_neg_1[0]
                        batch_size = input_neg_1.size(0)
                        optimizer_model.zero_grad()

                        # SASRec
                        # y_pred = model(input_t_1, input_pos_1, input_neg_1)
                        # .....
                        # other
                        input = torch.cat([input_1, input_neg_1], dim=0)
                        y_pred = model(input)
                        # ...
                        weight = torch.tensor([1.]).cuda()
                        weight = torch.repeat_interleave(weight, batch_size * 2, dim=0)
                        target = torch.tensor(np.concatenate((np.ones(batch_size), np.zeros(batch_size)))).cuda()
                        loss_base_model = base_model_cross_entropy(y_pred, target, weight=weight)

                        # SASRec
                        # pred_list = y_pred.tolist()
                        # .....
                        # other
                        pred_list = y_pred[:, 1].tolist()
                        # ...
                        target_list = target.tolist()
                        evaluate = Metric(pred_list, target_list)
                        train_epoch_base_loss += loss_base_model.tolist()
                        train_epoch_base_auc += evaluate.auc()

                        loss_base_model.backward()

                        optimizer_model.step()
                        # scheduler.step()

                    train_epoch_base_loss /= train_steps_per_epoch_1
                    train_epoch_base_auc /= train_steps_per_epoch_1

                    print(
                        f"Epoch {epoch}, train base loss {train_epoch_base_loss}, train base auc {train_epoch_base_auc}")
                    train_test_log.append(
                        f"Epoch {epoch}, train base loss {train_epoch_base_loss}, train base auc {train_epoch_base_auc}")

                for epoch in range(epochs_2):
                    train_epoch_base_loss = 0
                    train_epoch_base_auc = 0
                    # SASRec
                    # for input_t, input_pos, input_neg in zip(tqdm(trainloader_t), posloader, negloader):
                    #     input_t = input_t[0]
                    #     input_pos = input_pos[0]
                    # .....
                    # other
                    for input, input_neg in zip(tqdm(trainloader), trainloader_neg):
                        input = input[0]
                    # ...
                        input_neg = input_neg[0]
                        batch_size = input_neg.size(0)
                        optimizer_model.zero_grad()

                        # SASRec
                        # y_pred = model(input_t, input_pos, input_neg)
                        # .....
                        # other
                        input = torch.cat([input, input_neg], dim=0)
                        y_pred = model(input)
                        # ...
                        weight = torch.tensor([1.]).cuda()
                        weight = torch.repeat_interleave(weight, batch_size * 2, dim=0)
                        target = torch.tensor(np.concatenate((np.ones(batch_size), np.zeros(batch_size)))).cuda()
                        loss_base_model = base_model_cross_entropy(y_pred, target, weight=weight)

                        # SASRec
                        # pred_list = y_pred.tolist()
                        # .....
                        # other
                        pred_list = y_pred[:, 1].tolist()
                        # ...
                        target_list = target.tolist()
                        evaluate = Metric(pred_list, target_list)
                        train_epoch_base_loss += loss_base_model.tolist()
                        train_epoch_base_auc += evaluate.auc()

                        loss_base_model.backward()

                        optimizer_model.step()
                        # scheduler.step()

                    train_epoch_base_loss /= train_steps_per_epoch_1
                    train_epoch_base_auc /= train_steps_per_epoch_1

                    print(
                        f"Epoch {epoch}, train base loss {train_epoch_base_loss}, train base auc {train_epoch_base_auc}")
                    train_test_log.append(
                        f"Epoch {epoch}, train base loss {train_epoch_base_loss}, train base auc {train_epoch_base_auc}")

            with torch.no_grad():
                test_epoch_acc_5 = 0
                test_epoch_ndcg_5 = 0
                test_epoch_precision_5 = 0
                test_epoch_recall_5 = 0
                test_epoch_hr_5 = 0
                test_epoch_mrr_5 = 0
                test_epoch_loss, test_epoch_auc = 0, 0
                for input, target in tqdm(testloader):
                    # SASRec
                    # item = input[:, 39]
                    # input = input[0, 1:20]
                    # item = input[:, 100]
                    # input = input[0, 5:24]
                    # zero_indices = torch.where(input == 0)[0]
                    # input = torch.cat((input[zero_indices], input[input != 0]))
                    # y_pred = model.predict(input, item)
                    # .....
                    # other
                    y_pred = model(input)
                    # ...
                    batch_size = len(target)
                    weight = torch.tensor([1.]).cuda()
                    weight = torch.repeat_interleave(weight, batch_size, dim=0)
                    loss = base_model_cross_entropy(y_pred, target, weight=weight)
                    test_epoch_loss += loss.tolist()

                    pred_list = y_pred.tolist()
                    target_list = target.tolist()
                    # SASRec
                    # pred_list = [[x] for sublist in pred_list for x in sublist]
                    # .....
                    precision_5 = metrics.precision_n(pred_list, target_list, top_n=5)  # 默认是20
                    recall_5 = metrics.recall_n(pred_list, target_list, top_n=5)
                    acc_5 = metrics.acc_n(pred_list, target_list, top_n=5)
                    hr_5 = metrics.hr_n(pred_list, target_list, top_n=5)
                    mrr_5 = metrics.mrr_n(pred_list, target_list, top_n=5)
                    ndcg_5 = metrics.ndcg_n(pred_list, target_list, top_n=5)
                    test_epoch_precision_5 += precision_5
                    test_epoch_recall_5 += recall_5
                    test_epoch_acc_5 += acc_5
                    test_epoch_hr_5 += hr_5
                    test_epoch_mrr_5 += mrr_5
                    test_epoch_ndcg_5 += ndcg_5

                    # SASRec
                    # pred_list = y_pred[0].tolist()
                    # .....
                    # other
                    pred_list = y_pred[:, 1].tolist()
                    # ...
                    target_list = target.tolist()
                    evaluate = Metric(pred_list, target_list)
                    test_epoch_auc += evaluate.auc()

            test_epoch_loss /= test_steps_per_epoch
            test_epoch_auc /= test_steps_per_epoch
            test_epoch_ndcg_5 /= test_steps_per_epoch
            test_epoch_acc_5 /= test_steps_per_epoch
            test_epoch_hr_5 /= test_steps_per_epoch
            test_epoch_mrr_5 /= test_steps_per_epoch
            test_epoch_precision_5 /= test_steps_per_epoch
            test_epoch_recall_5 /= test_steps_per_epoch

            print(f"test loss {test_epoch_loss}, test auc {test_epoch_auc}, test acc_5 {test_epoch_acc_5}, "
                  f"test hr_5 {test_epoch_hr_5}, test mrr_5 {test_epoch_mrr_5}, test ndcg_5 {test_epoch_ndcg_5}, "
                  f"test precision_5 {test_epoch_precision_5}, test recall_5 {test_epoch_recall_5}")

            train_test_log.append(
                f"test loss {test_epoch_loss}, test auc {test_epoch_auc}, test acc_5 {test_epoch_acc_5}, "
                f"test hr_5 {test_epoch_hr_5}, test mrr_5 {test_epoch_mrr_5}, test ndcg_5 {test_epoch_ndcg_5}, "
                f"test precision_5 {test_epoch_precision_5}, test recall_5 {test_epoch_recall_5}")

            with open(log_name, "a") as f:
                for log in train_test_log:
                    f.write(log + "\n")


    run(dataset="amazon", split_n=2, test="amazon_data/amazon_test_data/amazon_test_data_split2.npy",
        update=True, batch_size=256, test_batch_size=64, shuffle=False, epochs_1=4, epochs_2=4,
        log_name="note/update_epochs8", con=True)

