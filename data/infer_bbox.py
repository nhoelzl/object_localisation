#!/usr/bin/env python3

# Infer bounding box coordinates from pose information
#
# 2021 (c) Micha Johannes Birklbauer
#
# https://github.com/t0xic-m/
# micha.birklbauer@gmail.com

import pandas as pd

relevant_cols = ["r ankle_X", "r ankle_Y", "r knee_X", "r knee_Y", "r hip_X",
                "r hip_Y", "l hip_X", "l hip_Y", "l knee_X", "l knee_Y",
                "l ankle_X", "l ankle_Y", "pelvis_X", "pelvis_Y", "thorax_X",
                "thorax_Y", "upper neck_X", "upper neck_Y", "head top_X",
                "head top_Y", "r wrist_X", "r wrist_Y", "r elbow_X",
                "r elbow_Y", "r shoulder_X", "r shoulder_Y", "l shoulder_X",
                "l shoulder_Y", "l elbow_X", "l elbow_Y", "l wrist_X",
                "l wrist_Y"]

def get_ymin(df_row):

    tmp_values = []

    for col in relevant_cols:
        if "Y" in col:
            if df_row[col] > 0:
                tmp_values.append(df_row[col])

    return min(tmp_values)

def get_ymax(df_row):

    tmp_values = []

    for col in relevant_cols:
        if "Y" in col:
            if df_row[col] > 0:
                tmp_values.append(df_row[col])

    return max(tmp_values)

def get_xmin(df_row):

    tmp_values = []

    for col in relevant_cols:
        if "X" in col:
            if df_row[col] > 0:
                tmp_values.append(df_row[col])

    return min(tmp_values)

def get_xmax(df_row):

    tmp_values = []

    for col in relevant_cols:
        if "X" in col:
            if df_row[col] > 0:
                tmp_values.append(df_row[col])

    return max(tmp_values)

if __name__ == "__main__":

    df = pd.read_csv("mpii_human_pose.csv")
    df["ymin"] = df.apply(lambda x: get_ymin(x), axis = 1)
    df["ymax"] = df.apply(lambda x: get_ymax(x), axis = 1)
    df["xmin"] = df.apply(lambda x: get_xmin(x), axis = 1)
    df["xmax"] = df.apply(lambda x: get_xmax(x), axis = 1)
    df.to_csv("mpii_human_pose_bbox.csv", index = False)
