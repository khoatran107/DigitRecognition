import numpy as np

def chuan_hoa(data):
    return data/255

def get_vector(data):
    data = chuan_hoa(data)
    data_shape = data.shape
    sample_count = data_shape[0]
    res = data.reshape((sample_count, data_shape[1] * data_shape[2]))
    return res
