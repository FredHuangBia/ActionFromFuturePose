'''
56,880 action clips in 
60 action classes
40 volunteers 
3 camera views 
25 joints 
at most 2 subjects
name format SsssCcccPpppRrrrAaaa (e.g. S001C002P003R002A013)
'''

import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap

from ntu_read_skeleton import read_xyz_fp

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body = 2
num_joint = 25
max_frame = 300
toolbar_width = 30

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(data_path,
            out_path,
            ignored_sample_path=None,
            benchmark='xview',
            part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    total_num_body = 0
    all_labels = []
    all_names = []
    all_num_bodys = []
    all_lengths = []
    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Consensus {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        data, num_body = read_xyz_fp(os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        total_num_body += num_body
        for body_id in range(num_body):
            all_names.append(s)
            all_labels.append(sample_label[i])
            all_num_bodys.append(num_body)
            all_lengths.append(data.shape[1])
    end_toolbar()

    with open('{}/{}_fp_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((all_names, all_labels, all_lengths, all_num_bodys), f)
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    fp = open_memmap(
        '{}/{}_fp_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(total_num_body, 3, max_frame, num_joint, 1))

    data_idx = 0
    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        data, num_body = read_xyz_fp(os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        for body_idx in range(num_body):
            fp[data_idx, :, 0:data.shape[1], :, 0] = data[:, :, :, body_idx]
            data_idx += 1
    end_toolbar()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='../../Datasets/NTU/skeletons')
    parser.add_argument(
        '--ignored_sample_path',
        default='../../Datasets/NTU/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='../../Datasets/NTU/NTU-RGB-D')

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)