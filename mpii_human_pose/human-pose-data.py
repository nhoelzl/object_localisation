"""
Created on 30.12.2020, by Nicole Hölzl
"""

# import packages
import os
import sys
import scipy.io as sio
import pandas as pd

# load mat data structure
loader = sio.loadmat('../data/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat',
                     struct_as_record=False)
loader = loader['RELEASE']
# noinspection PyProtectedMember
names = loader[0, 0]._fieldnames
print(names)

annolist = loader[0, 0].__dict__['annolist']
act = loader[0, 0].__dict__['act']
img_train = loader[0, 0].__dict__['img_train']

# define structure of dataframe
data_arr = ['NAME', 'r ankle_X', 'r ankle_Y', 'r knee_X', 'r knee_Y', 'r hip_X', 'r hip_Y', 'l hip_X', 'l hip_Y',
            'l knee_X', 'l knee_Y', 'l ankle_X', 'l ankle_Y', 'pelvis_X', 'pelvis_Y', 'thorax_X', 'thorax_Y',
            'upper neck_X', 'upper neck_Y', 'head top_X', 'head top_Y', 'r wrist_X', 'r wrist_Y', 'r elbow_X',
            'r elbow_Y', 'r shoulder_X', 'r shoulder_Y', 'l shoulder_X', 'l shoulder_Y', 'l elbow_X', 'l elbow_Y',
            'l wrist_X', 'l wrist_Y', 'Scale', 'Activity', 'Category']

# create dataframe
data = pd.DataFrame(columns=data_arr)

for ix in range(0, annolist.shape[1]):
    if img_train[0, ix] == 0:
        continue
    temp_arr = []
    obj_list = annolist[0, ix]
    obj_act = act[ix, 0]

    rect = obj_list.__dict__['annorect']
    img_d = obj_list.__dict__['image']
    if rect.shape[0] == 0:
        continue

    obj_rect = rect[0, 0]
    obj_img = img_d[0, 0]

    # noinspection PyProtectedMember
    if 'annopoints' not in obj_rect._fieldnames:
        continue

    name_d = obj_img.__dict__['name']
    name = name_d[0]
    temp_arr.append(name)
    annopoints = obj_rect.__dict__['annopoints']
    if annopoints.shape[0] == 0:
        continue
    obj_points = annopoints[0, 0]
    points = obj_points.__dict__['point']
    cnt = 0
    px = 0

    for n in range(0, 32):
        temp_arr.append(-1)

    for px in range(0, points.shape[1]):
        po = points[0, px]
        po_id = po.__dict__['id']

        po_x = po.__dict__['x']
        po_y = po.__dict__['y']
        ind = 2 * po_id[0][0] + 1
        temp_arr[ind] = po_x[0][0]
        temp_arr[ind + 1] = po_y[0][0]

    scale = obj_rect.__dict__['scale']
    temp_arr.append(scale[0][0])

    activity = act[ix, 0]

    a_n = activity.act_name
    c_n = activity.cat_name

    if a_n.shape[0] == 0:
        temp_arr.append(a_n)
    else:
        temp_arr.append(activity.act_name[0])
    if c_n.shape[0] == 0:
        temp_arr.append(c_n)
    else:
        temp_arr.append(activity.cat_name[0])

    temp_data_f = pd.DataFrame([temp_arr], columns=data_arr)

    data = pd.concat([data, temp_data_f])

    if ix % 100 == 0:
        print(ix)

data.index = [x for x in range(1, len(data.values)+1)]
data.index.name = 'ID'

# save as csv in data folder
data.to_csv('../data/mpii_human_pose.csv')
