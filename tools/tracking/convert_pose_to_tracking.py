from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import ujson as json
import os
import cv2
import numpy as np
from tqdm import tqdm

import codecs
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


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '--outdir', help="output dir for json files",
        required=True, type=str)
    parser.add_argument(
        '--datadir', help="keypoint file",
        required=True, type=str)
    parser.add_argument(
        '--keypoint-file', help="keypoint file",
        required=True, type=str)

    return parser.parse_args()


def load_images(image_root):
    image_filenames = []
    curr_dir = os.path.abspath(os.curdir)
    os.chdir(image_root)
    for root, dirs, filenames in os.walk('.'):
        for filename in sorted(filenames):
            name, ext = os.path.splitext(filename)
            if ext in {'.jpg', '.png'}:
                image_filenames.append(os.path.join(root, filename))
    os.chdir(curr_dir)
    return image_filenames


def get_dummy_detection(keypoints, scores, bbox_ltwh):
    score = bbox_ltwh[4]
    return {
        'score': score,
        'bbox_ltwh': np.asarray(bbox_ltwh[:4], dtype=np.int32),
        'keypoints': np.asarray(keypoints, dtype=np.float32),
        'keypoints_score': np.asarray(scores, dtype=np.float32).reshape(-1),
    }


def convert_tracking(data_dir, out_dir, keypoint_file, min_score=0.4):
    image_root = os.path.join(data_dir, 'frames')
    with open(keypoint_file, 'r') as f:
        pose_data = json.load(f)

    # index keypoint
    camera_time_to_data = {}
    for filename, data in tqdm(pose_data.items(), total=len(pose_data)):
        name = os.path.splitext(filename)[0]
        camera_id, timestamp = os.path.split(name)
        camera_id = os.path.basename(camera_id)

        camera_time_to_data[(camera_id, timestamp)] = data

    # index images
    image_names = load_images(image_root)
    for image_filename in tqdm(image_names, total=len(image_names)):
        name = os.path.splitext(image_filename)[0]
        camera_id, timestamp = os.path.split(name)
        camera_id = os.path.basename(camera_id)

        image = cv2.imread(os.path.join(image_root, image_filename))
        image_wh = (image.shape[1], image.shape[0])

        file_data = {
            'camera_id': int(camera_id),
            'timestamp': float(timestamp),
            'image_wh': image_wh,
            'people': []
        }
        output_file = os.path.join(out_dir, name + '.json')

        data = camera_time_to_data.get((camera_id, timestamp), None)
        if data is not None:
            tlwhs = np.asarray(data['boxes'], dtype=np.float32)
            keypoints = np.asarray(data['keypoints'], dtype=np.float32)
            for tlwh, keypoint in zip(tlwhs, keypoints):
                keypoints_2d = keypoint[:, 0:2]
                scores = keypoint[:, 2]
                person = get_dummy_detection(keypoints_2d, scores, tlwh)
                if person['score'] < min_score:
                    continue
                file_data['people'].append(person)

        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        save(file_data, output_file)


if __name__ == '__main__':
    args = parse_args()
    convert_tracking(args.datadir, args.outdir, args.keypoint_file)
