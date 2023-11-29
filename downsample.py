import numpy as np
from tqdm import tqdm

def chuan_hoa(data):
    return data/255

def get_downsample(data):
    data = chuan_hoa(data)
    data_shape = data.shape
    sample_count = data_shape[0]
    new_shape = (sample_count, data_shape[1] // 2, data_shape[2] // 2)
    res = np.empty(new_shape)
    for sample_index in tqdm(range(sample_count)):
        for i in range(new_shape[1]):
            for j in range(new_shape[2]):
                res[sample_index, i, j] = np.average(data[sample_index,
                                                          2*i : 2*i+2, 2*j :
                                                          2*j+2])
    return res.reshape((sample_count, new_shape[1] * new_shape[2]))
