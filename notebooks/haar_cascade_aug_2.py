#!/usr/bin/env python3

# 2021 (c) Micha Johannes Birklbauer
#
# https://github.com/t0xic-m/
# micha.birklbauer@gmail.com

import numpy as np
import cv2
import os
import subprocess
import fileinput
import time


# path definitions
pp = "D:\\Users\\Micha\\Documents\\GitHub\\object_localisation\\notebooks\\" # project_path
trainImgPathPos = pp + "haar_cascade_aug\\refImages_raw"
# false images taken from:
# https://ai.vub.ac.be/
# http://arti.vub.ac.be/research/colour/data/imagesets.zip
trainImgPathNeg = pp + "haar_cascade_aug\\false_images"

openCVbinPath = pp + "haar_cascade\\opencv\\build\\x64\\vc15\\bin"
openCVAnnotationToolExe = "opencv_annotation.exe"
openCVSamplesToolExe = "opencv_createsamples.exe"
openCVTrainCascadeToolExe = "opencv_traincascade.exe"
cascadeResultPath = pp + "haar_cascade_aug\\trained_cascade"

tgtFileExtensionTuple = ext = [".JPEG", ".jpg", ".png"]
fileDictName = "out.txt"
annotationFileName = "annotation.txt"
vectorFileName = "samples.vect"

def prepareSamplesAsVectFile(annotationFilePath, bgImages, vectorFileName, doShow = False) :
    prepareSamplesCommand = os.path.join(openCVbinPath, openCVSamplesToolExe)
    vectorFile = os.path.join(annotationFilePath, vectorFileName)
    annotationFile = os.path.join(annotationFilePath, annotationFileName)
    numVal = 500
    widthVal = 32
    heightVal = 64
    bgThreshVal = 0
    maxAngleVal = 0
    prepareSamplesCommand = prepareSamplesCommand + " -vec " + str(vectorFile) + " -info " + \
    str(annotationFile) + " -bg " + str(bgImages) + " -num " + str(numVal) + " -w " + str(widthVal) + " -h " + str(heightVal)
    if doShow:
        prepareSamplesCommand = prepareSamplesCommand + " -show"
    print("trying to call command: " + prepareSamplesCommand)
    os.system(prepareSamplesCommand)

def trainHaarCascade(cascadeResultPath, annotationFilePath, vectorFileName, bgImages) :
    vectorFile = os.path.join(annotationFilePath, vectorFileName)
    prepareSamplesCommand = os.path.join(openCVbinPath, openCVTrainCascadeToolExe)
    widthVal = 32
    heightVal = 64
    numStages = 30
    numPositives = 500
    numNegatives = 200
    maxFalseAlarmRate = 0.6

    prepareSamplesCommand = prepareSamplesCommand + " -data " + str(cascadeResultPath) + " -vec " + str(vectorFile) + " -bg " + \
    str(bgImages) + " -w " + str(widthVal) + " -h " + str(heightVal) + " -numStages " + str(numStages) + " -numPos " + \
    str(numPositives) + " -numNeg " + str(numNegatives) + " -maxFalseAlarmRate " + str(maxFalseAlarmRate)
    print("trying to call command: " + prepareSamplesCommand)
    os.system(str(prepareSamplesCommand))
    input("Press Enter to continue...")

trainHaarCascade(cascadeResultPath, trainImgPathPos, vectorFileName, os.path.join(trainImgPathNeg, fileDictName))
