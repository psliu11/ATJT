import torch
import torch.nn as nn
import torch.nn.functional as F


class DIN(nn.Module):
    def __init__(self, user_count, item_count, cate_count, max_sl):
        super(DIN, self).__init__()

        self.attention_mlp = nn.Linear(64, 32)
        self.attention_classify = nn.Linear(32, 1)

        self.dnn_1 = nn.Linear(40, 256)
        self.dnn_2 = nn.Linear(256, 64)
        self.classifier = nn.Linear(64, 2)

        self.dropout = nn.Dropout(p=0.5)

        self.embedding_dict = nn.ModuleDict()
        self.embedding_dict["user_id"] = nn.Embedding(user_count, 8)
        self.embedding_dict["item"] = nn.Embedding(item_count, 8)
        self.embedding_dict["cate"] = nn.Embedding(cate_count, 8)

        self.max_sl = torch.from_numpy(max_sl).cuda()

        self.sigmoid = nn.Sigmoid()

        # 初始化权重
        for m in self.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
                torch.nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x):
        max_sl = 0
        for i in x[:, self.max_sl * 2 + 3]:
            max_sl = max(max_sl, i)

        # embed层
        user_id_emb = self.embedding_dict["user_id"](x[:, 0])
        hist_item_emb = self.embedding_dict["item"](x[:, 1: max_sl + 1])
        hist_cate_emb = self.embedding_dict["cate"](x[:, self.max_sl + 1: self.max_sl + max_sl + 1])
        item_emb = self.embedding_dict["item"](x[:, self.max_sl * 2 + 1])
        cate_emb = self.embedding_dict["cate"](x[:, self.max_sl * 2 + 2])

        # item和cate拼接
        hist_emb = torch.cat([hist_item_emb, hist_cate_emb], dim=2)  # [batch, self.max-1, dim]
        target_emb = torch.cat([item_emb, cate_emb], dim=1).unsqueeze(1)  # [batch, 1, dim]
        target_emb_repeat = torch.repeat_interleave(target_emb, max_sl, dim=1)  # [batch, self.max-1, dim]
        attention_input = torch.cat(
            [hist_emb, target_emb_repeat, hist_emb - target_emb_repeat, hist_emb * target_emb_repeat], dim=2)

        attention_out = self.attention_mlp(attention_input)
        # if dropout:
        #     out = self.dropout(out)
        attention_out = self.dice(attention_out, attention_out.size(2), dim=3)
        weights = self.attention_classify(attention_out).squeeze(2)  # [B, T]

        keys_masks = x[:, 1: max_sl + 1] > 0  # [B, T]
        keys_masks = keys_masks.float()  # [B, T]

        exp_weights = torch.exp(weights)
        masked_exp_weights = exp_weights * keys_masks
        weights = masked_exp_weights / masked_exp_weights.sum(1).unsqueeze(1)

        hist_emb_ = (weights.unsqueeze(2) * hist_emb).sum(1)
        attention_emb = torch.cat([user_id_emb, hist_emb_, target_emb.squeeze(1)], dim=1)

        out = self.dnn_1(attention_emb)
        # if dropout:
        #     out = self.dropout(out)
        out = self.dice(out, out.size(1))
        out = self.dnn_2(out)
        # if dropout:
        #     out = self.dropout(out)
        out = self.dice(out, out.size(1))
        out = self.classifier(out)

        return F.softmax(out, dim=1)

    def dice(self, x, emb_size, dim=2, epsilon=1e-8):
        bn = nn.BatchNorm1d(emb_size, eps=epsilon).cuda()
        if dim == 2:
            alpha = nn.Parameter(torch.zeros((emb_size,)).cuda())
            x_p = self.sigmoid(bn(x))
            out = alpha * (1 - x_p) * x + x_p * x

        if dim == 3:
            alpha = nn.Parameter(torch.zeros((emb_size, 1)).cuda())
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(bn(x))
            out = alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)

        return out


class DIN_1(nn.Module):
    def __init__(self, user_count, item_count, cate_count, max_sl, city_count, postal_count, stars_count, useful_count,
                 cool_count, funny_count, average_stars_count):
        super(DIN_1, self).__init__()

        self.attention_mlp = nn.Linear(160, 32)
        self.attention_classify = nn.Linear(32, 1)

        self.dnn_1 = nn.Linear(120, 256)
        self.dnn_2 = nn.Linear(256, 64)
        self.classifier = nn.Linear(64, 2)

        self.dropout = nn.Dropout(p=0.5)

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

        self.sigmoid = nn.Sigmoid()

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

        # item和cate拼接
        hist_emb = torch.cat([hist_item_emb, hist_cate_emb, hist_city_emb, hist_postal_emb, hist_stars_emb], dim=2)  # [batch, self.max-1, dim]
        target_emb = torch.cat([item_emb, cate_emb, city_emb, postal_emb, stars_emb], dim=1).unsqueeze(1)  # [batch, 1, dim]
        target_emb_repeat = torch.repeat_interleave(target_emb, max_sl, dim=1)  # [batch, self.max-1, dim]
        attention_input = torch.cat(
            [hist_emb, target_emb_repeat, hist_emb - target_emb_repeat, hist_emb * target_emb_repeat], dim=2)

        attention_out = self.attention_mlp(attention_input)
        # if dropout:
        #     out = self.dropout(out)
        attention_out = self.dice(attention_out, attention_out.size(2), dim=3)
        weights = self.attention_classify(attention_out).squeeze(2)  # [B, T]

        keys_masks = x[:, 1: max_sl + 1] > 0  # [B, T]
        keys_masks = keys_masks.float()  # [B, T]

        exp_weights = torch.exp(weights)
        masked_exp_weights = exp_weights * keys_masks
        weights = masked_exp_weights / masked_exp_weights.sum(1).unsqueeze(1)

        hist_emb_ = (weights.unsqueeze(2) * hist_emb).sum(1)
        attention_emb = torch.cat([user_id_emb, useful_emb, cool_emb, funny_emb, average_stars_emb, hist_emb_,
                                   target_emb.squeeze(1)], dim=1)

        out = self.dnn_1(attention_emb)
        # if dropout:
        #     out = self.dropout(out)
        out = self.dice(out, out.size(1))
        out = self.dnn_2(out)
        # if dropout:
        #     out = self.dropout(out)
        out = self.dice(out, out.size(1))
        out = self.classifier(out)

        return F.softmax(out, dim=1)

    def dice(self, x, emb_size, dim=2, epsilon=1e-8):
        bn = nn.BatchNorm1d(emb_size, eps=epsilon).cuda()
        if dim == 2:
            alpha = nn.Parameter(torch.zeros((emb_size,)).cuda())
            x_p = self.sigmoid(bn(x))
            out = alpha * (1 - x_p) * x + x_p * x

        if dim == 3:
            alpha = nn.Parameter(torch.zeros((emb_size, 1)).cuda())
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(bn(x))
            out = alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)

        return out