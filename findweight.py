import torch
import torch.nn as nn


class FindWeight(nn.Module):
    def __init__(self, user_count, item_count, cate_count, max_sl):
        super(FindWeight, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.weight_attention_mlp1 = nn.Linear(64, 32)  # 注意力
        self.weight_attention_classify = nn.Linear(32, 1)  # 注意力
        self.dnn_1 = nn.Linear(40, 256)
        self.dnn_2 = nn.Linear(256, 64)
        self.batch_weight_attention = nn.Linear(64, 1)

        self.weight_1 = nn.Linear(1, 1)

        self.embedding_dict = nn.ModuleDict()
        self.embedding_dict["user_id"] = nn.Embedding(user_count, 8)
        self.embedding_dict["item"] = nn.Embedding(item_count, 8)
        self.embedding_dict["cate"] = nn.Embedding(cate_count, 8)

        self.max_sl = torch.from_numpy(max_sl).cuda()

        # 初始化权重
        for m in self.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
                torch.nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x):
        max_sl = 0
        for i in x[:, self.max_sl * 2 + 3]:
            max_sl = max(max_sl, i)

        # embed层
        user_id_emb = self.embedding_dict["user_id"](x[:, 0]).clone()
        hist_item_emb = self.embedding_dict["item"](x[:, 1: max_sl + 1]).clone()
        hist_cate_emb = self.embedding_dict["cate"](x[:, self.max_sl + 1: self.max_sl + max_sl + 1]).clone()
        item_emb = self.embedding_dict["item"](x[:, self.max_sl * 2 + 1]).clone()
        cate_emb = self.embedding_dict["cate"](x[:, self.max_sl * 2 + 2]).clone()

        hist_emb = torch.cat([hist_item_emb, hist_cate_emb], dim=2)  # [batch, self.max-1, dim]
        target_emb = torch.cat([item_emb, cate_emb], dim=1).unsqueeze(1)  # [batch, 1, dim]
        target_emb_repeat = torch.repeat_interleave(target_emb, max_sl, dim=1)

        keys_masks = x[:, 1: max_sl + 1] > 0  # [B, T]
        keys_masks = keys_masks.float()

        # 注意力
        weight_attention_input = torch.cat(
            [hist_emb, target_emb_repeat, hist_emb - target_emb_repeat, hist_emb * target_emb_repeat], dim=2)
        weight_attention_out = self.weight_attention_mlp1(weight_attention_input)
        weight_attention_out = self.dropout(weight_attention_out)
        weight_attention_out = self.prelu(weight_attention_out)
        weight_weights = self.weight_attention_classify(weight_attention_out).squeeze(2)
        batch_weight_masked_weights = weight_weights * keys_masks
        hist_emb_ = (batch_weight_masked_weights.unsqueeze(2) * hist_emb).sum(1)

        attention_emb = torch.cat([user_id_emb, hist_emb_, target_emb.squeeze(1)], dim=1)
        out1 = self.dnn_1(attention_emb)
        out1 = self.prelu(out1)
        out2 = self.dnn_2(out1)
        out2 = self.prelu(out2)
        click_batch_weight = self.batch_weight_attention(out2)
        click_batch_weight = self.sigmoid(click_batch_weight)

        return click_batch_weight.squeeze(1)


class FindWeight_1(nn.Module):
    def __init__(self, user_count, item_count, cate_count, max_sl, city_count, postal_count, stars_count, useful_count,
                 cool_count, funny_count, average_stars_count):
        super(FindWeight_1, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.weight_attention_mlp1 = nn.Linear(160, 32)  # 注意力
        self.weight_attention_classify = nn.Linear(32, 1)  # 注意力
        self.dnn_1 = nn.Linear(120, 256)
        self.dnn_2 = nn.Linear(256, 64)
        self.batch_weight_attention = nn.Linear(64, 1)

        self.embedding_dict = nn.ModuleDict()
        self.embedding_dict["user_id"] = nn.Embedding(user_count, 8)
        self.embedding_dict["item"] = nn.Embedding(item_count, 8)
        self.embedding_dict["cate"] = nn.Embedding(cate_count, 8)
        self.embedding_dict["city"] = nn.Embedding(city_count, 8)
        self.embedding_dict["postal"] = nn.Embedding(postal_count, 8)
        self.embedding_dict["stars"] = nn.Embedding(stars_count, 8)
        self.embedding_dict["useful"] = nn.Embedding(useful_count, 8)
        self.embedding_dict["cool"] = nn.Embedding(cool_count, 8)
        self.embedding_dict["funny"] = nn.Embedding(funny_count, 8)
        self.embedding_dict["average_stars"] = nn.Embedding(average_stars_count, 8)

        self.max_sl = torch.from_numpy(max_sl).cuda()

        # 初始化权重
        for m in self.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
                torch.nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x):
        max_sl = 0
        for i in x[:, self.max_sl * 5 + 10]:
            max_sl = max(max_sl, i)

        # embed层
        user_id_emb = self.embedding_dict["user_id"](x[:, 0])
        useful_emb = self.embedding_dict["useful"](x[:, 1])
        funny_emb = self.embedding_dict["funny"](x[:, 2])
        cool_emb = self.embedding_dict["cool"](x[:, 3])
        average_stars_emb = self.embedding_dict["average_stars"](x[:, 4])
        hist_item_emb = self.embedding_dict["item"](x[:, 5: max_sl + 5])
        hist_cate_emb = self.embedding_dict["cate"](x[:, self.max_sl + 5: self.max_sl + max_sl + 5])
        hist_city_emb = self.embedding_dict["city"](x[:, 2 * self.max_sl + 5: 2 * self.max_sl + max_sl + 5])
        hist_postal_emb = self.embedding_dict["postal"](x[:, 3 * self.max_sl + 5: 3 * self.max_sl + max_sl + 5])
        hist_stars_emb = self.embedding_dict["stars"](x[:, 4 * self.max_sl + 5: 4 * self.max_sl + max_sl + 5])
        item_emb = self.embedding_dict["item"](x[:, self.max_sl * 5 + 5])
        cate_emb = self.embedding_dict["cate"](x[:, self.max_sl * 5 + 6])
        city_emb = self.embedding_dict["city"](x[:, self.max_sl * 5 + 7])
        postal_emb = self.embedding_dict["postal"](x[:, self.max_sl * 5 + 8])
        stars_emb = self.embedding_dict["stars"](x[:, self.max_sl * 5 + 9])

        hist_emb = torch.cat([hist_item_emb, hist_cate_emb, hist_city_emb, hist_postal_emb, hist_stars_emb], dim=2)  # [batch, self.max-1, dim]
        target_emb = torch.cat([item_emb, cate_emb, city_emb, postal_emb, stars_emb], dim=1).unsqueeze(1)  # [batch, 1, dim]
        target_emb_repeat = torch.repeat_interleave(target_emb, max_sl, dim=1)

        keys_masks = x[:, 1: max_sl + 1] > 0  # [B, T]
        keys_masks = keys_masks.float()

        # 注意力
        weight_attention_input = torch.cat(
            [hist_emb, target_emb_repeat, hist_emb - target_emb_repeat, hist_emb * target_emb_repeat], dim=2)
        weight_attention_out = self.weight_attention_mlp1(weight_attention_input)
        weight_attention_out = self.dropout(weight_attention_out)
        weight_attention_out = self.prelu(weight_attention_out)
        weight_weights = self.weight_attention_classify(weight_attention_out).squeeze(2)
        batch_weight_masked_weights = weight_weights * keys_masks
        hist_emb_ = (batch_weight_masked_weights.unsqueeze(2) * hist_emb).sum(1)

        attention_emb = torch.cat([user_id_emb, useful_emb, cool_emb, funny_emb, average_stars_emb, hist_emb_,
                                   target_emb.squeeze(1)], dim=1)
        out1 = self.dnn_1(attention_emb)
        out1 = self.prelu(out1)
        out2 = self.dnn_2(out1)
        out2 = self.prelu(out2)
        click_batch_weight = self.batch_weight_attention(out2)
        click_batch_weight = self.sigmoid(click_batch_weight)

        return click_batch_weight.squeeze(1)
