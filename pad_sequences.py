def pad_sequences(x, maxlen):
    for i in range(maxlen):
        if i + 1 > len(x):
            x.append(0)
    return x