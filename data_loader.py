"""
Data loader

License is from https://github.com/aboulch/tec_prediction
"""

import torch.utils.data as data

# import skimage.io
import numpy as np
from PIL import Image, ImageMath
import os
import os.path
import h5py
from datetime import date, timedelta
import logging
import random

def make_dataset_train(seq_length):
    start_train = date(2014,1,1)
    end_train = date(2016,5,31)
    days = []
    day = start_train
    # break if no possibility to get the next sequence
    while (day < end_train - timedelta(days=(seq_length) // 12)):
        for k in range(12): # number of images per day
            days.append([day,k])
        day += timedelta(days=1)
    return days

def make_dataset_val(seq_length):
    start_val = date(2016,7,1)
    end_val = date(2016,12,31)
    days = []
    day = start_val
    # break if no possibility to get the next sequence
    while (day < end_val - timedelta(days=(seq_length) // 12)):
        for k in range(12): # number of images per day
            days.append([day,k])
        day += timedelta(days=1)
    return days


class SequenceLoader(data.Dataset):
    """Main Class for Image Folder loader."""

    def __init__(self, root_dir, seqLength, training=True):
        """Init function."""
        #
        # get the lists of images

        self.TEC_MAP_SHAPE = (72,72)
        self.seqLength = seqLength
        self.root_dir = root_dir
        if training:
            self.days = make_dataset_train(seqLength)
        else:
            self.days = make_dataset_val(seqLength)
        self.training = training


    def load(self, index):
        day_start, hour_start = self.days[index]

        day = day_start
        k = hour_start
        pos = 0
        
        dat = np.zeros((self.seqLength, 1,self.TEC_MAP_SHAPE[0], self.TEC_MAP_SHAPE[1]))

        for pos in range(self.seqLength):
            if k>=12:
                day += timedelta(days=1)
                k=0
            year = day.year
            day_of_year = day.timetuple().tm_yday
            try:
                tecdata = np.load(os.path.join(self.root_dir, "tecdata_{0:04d}_{1:03d}.npy".format(year, day_of_year)))
            except:
                if self.training:
                    return self.load(random.randint(0, self.__len__()-1))
                else:
                    print("Error validation set")
            dat[pos,0,:,:] = np.array(Image.fromarray(tecdata[k % 12,:,:]).resize((self.TEC_MAP_SHAPE[1], self.TEC_MAP_SHAPE[0])))
            k+=1

        return dat, np.array([index])

    def __getitem__(self, index):
        """Get item."""
        return self.load(index)
        
    def __len__(self):
        """Length."""
        return len(self.days)
