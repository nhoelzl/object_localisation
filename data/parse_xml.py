#!/usr/bin/env python3

# Read XML files and save values in CSV
#
# 2021 (c) Micha Johannes Birklbauer
#
# https://github.com/t0xic-m/
# micha.birklbauer@gmail.com

import os
import xmltodict
import pandas as pd

def parse_xml(dirname):
    imagenames = []
    ymin_values = []
    ymax_values = []
    xmin_values = []
    xmax_values = []
    for dirpath, dirnames, filenames in os.walk(dirname):
        for fname in filenames:
            if fname.split(".")[-1] == "xml":
                with open(os.path.join(dirpath, fname), "r") as f:
                    xml_data = f.read()
                    dict_data = xmltodict.parse(xml_data)
                    imagenames.append(dict_data["annotation"]["filename"])
                    ymin_values.append(float(dict_data["annotation"]["object"]["bndbox"]["ymin"]))
                    ymax_values.append(float(dict_data["annotation"]["object"]["bndbox"]["ymax"]))
                    xmin_values.append(float(dict_data["annotation"]["object"]["bndbox"]["xmin"]))
                    xmax_values.append(float(dict_data["annotation"]["object"]["bndbox"]["xmax"]))
                    f.close()

    df = pd.DataFrame({"NAME": imagenames, "ymin": ymin_values, "ymax": ymax_values, "xmin": xmin_values, "xmax": xmax_values})
    df.to_csv(dirname + ".csv", index = False)
    print(df.shape)
    return df

if __name__ == "__main__":

    df_1 = parse_xml("samples_singleperson")
    df_2 = parse_xml("samples_test")
