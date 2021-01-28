#!/usr/bin/env python3

# Data Sampling for manual annotation
#
# 2021 (c) Micha Johannes Birklbauer
#
# https://github.com/t0xic-m/
# micha.birklbauer@gmail.com

import os
import random
from shutil import copyfile

if __name__ == "__main__":

    random.seed(1337)

    dirname = "images"
    images = []
    for dirpath, dirnames, filenames in os.walk(dirname):
        for fname in filenames:
            images.append(os.path.join(dirpath, fname).replace("\\", "/"))

    print(len(images))

    samples = random.sample(images, 500)

    print(len(samples))

    os.mkdir("samples")

    for sample in samples:
        copyfile(sample, sample.replace("images", "samples"))
