import codecs
import numpy as np
import json
import copy


def np_encode(np_array):
    return {
        "np_array": codecs.decode(codecs.encode(np_array.tostring(), "base64"), "utf-8"),
        "shape": list(np_array.shape),
        "dtype": str(np_array.dtype)
    }


def np_decode(data):
    np_array = np.frombuffer(codecs.decode(codecs.encode(data['np_array'], 'utf-8'), 'base64'), dtype=data['dtype'])
    np_array = np_array.reshape(data['shape'])
    return np_array


def process(data, mode):
    '''Recursively traverse the data and decode/encode every numpy array, depending on 'mode'.

    Args:
        data: Data to be processed.
        mode (str): 'encode' or 'decode'.

    Returns:
        data: Encoded or decoded data, depending on 'mode'.
    '''

    if mode == 'encode' and isinstance(data, np.ndarray):
        data = np_encode(data)
    elif mode == 'decode' and isinstance(data, dict) and 'np_array' in data:
        data = np_decode(data)
    elif isinstance(data, (np.int64, np.int, np.int32)):
        data = int(data)
    elif isinstance(data, (np.float64, np.float, np.float32)):
        data = float(data)
    elif isinstance(data, list):
        data = [process(sub_data, mode) for sub_data in data]
    elif isinstance(data, dict):
        for key, sub_data in data.items():
            data[key] = process(sub_data, mode)
    return data


def load(filename):
    '''Load data from a json file and decode it.

    Args:
        filname (str): File to load data from.
    Returns:
        data: Data from file.
    '''
    with open(filename, 'r') as f:
        data = json.load(f)
    data = process(data, 'decode')
    return data


def save(data, filename):
    '''Encode data and save to a json file.

    Args:
        data: Data to save.
        filname (str): File to save data to.
    '''
    data = copy.deepcopy(data)
    data = process(data, 'encode')
    with open(filename, 'w') as f:
        json.dump(data, f)
