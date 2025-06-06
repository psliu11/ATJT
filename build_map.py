def build_map(df, col_name):
    """
    制作一个映射，键为列名，值为序列数字
    :param df: reviews_df / meta_df
    :param col_name: 列名
    :return: 字典，键
    """
    key = df[col_name].unique().tolist()
    key = sorted(['0' if x is None else x for x in key])
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key