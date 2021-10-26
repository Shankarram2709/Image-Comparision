#!/usr/bin/python3
import os
import sys
import tarfile
import argparse
import glob
import numpy as np
import pandas as pd

parser               = argparse.ArgumentParser()
parser.add_argument('-t','--tar-files',     dest='tar_file',     type=str, help='path to tar files containing input data.',  required=True)
parser.add_argument('-i','--input-path',     dest='input_path',     type=str, help='path to image input files.',  required=True)
parser.add_argument('-o','--output-path',     dest='output_lst',     type=str, help='path to output lst file.',  required=True)
args = parser.parse_args()

inpath = args.input_path
outpath = args.output_lst

if len(glob.glob(inpath+'/**/*.png', recursive=True))<=0:
    file = tarfile.open(args.tar_file)
    file.extractall(inpath)
    file.close()

if not os.path.isdir(inpath) or not os.access(inpath, os.R_OK):
        print("input dir does not exist or is not readable: {}. Creating".format(inpath))
        os.makedirs(inpath)

all_datapoint_paths = glob.glob(inpath+'/**/*.png', recursive=True)
all_datapoint_paths = [os.path.abspath(p) for p in all_datapoint_paths]
#generate random
all_datapoint_paths = np.random.permutation(all_datapoint_paths)

#for filePath in all_datapoint_paths:
#    print("{}".format(filePath))
datapoint_df = pd.DataFrame({'path': all_datapoint_paths})
datapoint_df.to_csv(outpath+'/'+'path.lst',index=False)