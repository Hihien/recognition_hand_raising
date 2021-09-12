import hashlib
import os
import pathlib
import shutil
import sys
import time
import cv2
import torch
import numpy as np
import json

# data_dir = 'D:/code/video-to-pose3D/'
# subsets = ['mica_train2021.json']
#
# for subset in subsets:
#     file_dir = os.path.join(data_dir, subset)
#     if not file_dir.endswith('.json'):
#         continue
#     with open(file_dir, 'r') as json_file:
#         annot = json.load(json_file)
#         for subject in annot['images']:
#             file_name = subject['file_name']
#             file_name = '/content/gdrive/MyDrive/Mica_150821/Mica_150821/img_train2021/' + file_name
#             subject['file_name'] = file_name
#
#             with open(file_dir, 'w') as f:
#                 f.write(json.dumps(annot))

data_dir = "D:/code/pose_3d_human/outputs/alpha_pose_1/split_image/"
output_path = 'D:/code/pose_3d_human/outputs/alpha_pose_1/'
file = 'D:/code/pose_3d_human/outputs/alpha_pose_1/alphapose-results.json'
kpts = []

coco_keypoints = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
left_person_inds = [coco_keypoints.index(_) for _ in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_hip', 'right_hip']]
right_person_inds = [coco_keypoints.index(_) for _ in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear','right_shoulder', 'right_elbow', 'right_wrist', 'left_hip', 'right_hip']]
j =0
str = ''
with open(file, 'r') as json_file:
    data = json.load(json_file)
    for i in range(len(data)):
        file_name = data[i]['image_id']
        file_name = data_dir + file_name
        coor = np.asarray(data[i]['keypoints']).reshape(-1, 3)
        xy = coor[:, :-1].round().astype(float)
        conf = coor[:, -1]

        left_person = xy[left_person_inds]
        right_person = xy[right_person_inds]

        str += f'{file_name} {left_person[0][0]} {left_person[0][1]} {left_person[1][0]} {left_person[1][1]} {left_person[2][0]} {left_person[2][1]} {left_person[3][0]} {left_person[3][1]} {left_person[4][0]} {left_person[4][1]} {left_person[5][0]} {left_person[5][1]} {left_person[6][0]} {left_person[6][1]} {left_person[7][0]} {left_person[7][1]} {left_person[8][0]} {left_person[8][1]} {left_person[9][0]} {left_person[9][1]}'
        str += '\n'
        str += f'{file_name} {right_person[0][0]} {right_person[0][1]} {right_person[1][0]} {right_person[1][1]} {right_person[2][0]} {right_person[2][1]} {right_person[3][0]} {right_person[3][1]} {right_person[4][0]} {right_person[4][1]} {right_person[5][0]} {right_person[5][1]} {right_person[6][0]} {right_person[6][1]} {right_person[7][0]} {right_person[7][1]} {right_person[8][0]} {right_person[8][1]} {right_person[9][0]} {right_person[9][1]}'
        str += '\n'
            # kpts.append(left_person)
            # kpts.append(right_person)
            #
            # kpts = np.array(kpts).astype(np.float32)}



    # name = f'{output_path}/video_1.npz'
    # kpts = np.array(kpts).astype(np.float32)
    # print('kpts npz save in ', name)
    # np.savez_compressed(name, kpts= kpts)


    name = f'{output_path}/video_1.txt'
    with open(name, 'a') as r:
        r.write(str)
        r.close()
#
# a = np.load('D:/code/pose_3d_human/outputs/alpha_pose_1//video_1.npz')
# print(a['kpts'])




