import os
import sys
import cv2
import torch
import numpy as np
from pprint import pprint
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
import shelve
from natsort import natsorted
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from PIL import Image


class ActionNormalizedValve(VisionDataset):
    ''' Sprite Dataset
        - sequence
            - train: 2000
            - test : 200
        - step              : 25
        - image size        : (3, 64, 64)
        - action variation  : 8
            - claw1だけ動く
            - claw2だけ動く
            - claw3だけ動く
            - ３本同時に動く（左右）など
        - minmax value:
            - min: 0
            - max: 255
    '''

    def __init__(self,
            img_dir  : str,
            train    : bool,
            transform: Optional[Callable] = None,
        ):
        self.train     = train
        self.transform = transform
        self.img_paths = self._get_img_paths(img_dir)
        self.num_data  = len(self.img_paths)


    def _get_img_paths(self, img_dir):
        """
        指定したディレクトリ内の画像ファイルのパス一覧を取得する。
        """
        if self.train: img_dir = img_dir + "/ActionNormalizedValve/dataset_202210221514_valve2000_train/"
        else         : img_dir = img_dir + "/ActionNormalizedValve/dataset_20221022153117_valve200_test/"
        img_dir = Path(img_dir)
        img_paths = [p for p in img_dir.iterdir() if p.suffix == ".db"]
        img_paths = natsorted(img_paths)
        # import ipdb; ipdb.set_trace()
        return img_paths


    def __getitem__(self, index: int):
        path                = self.img_paths[index]                      # 絶対パス: Ex.) PosixPath('data/Sprite/lpc-dataset/train/1808.sprite')
        path_without_suffix = str(path.resolve()).split(".")[0]          #; print(path_without_suffix)
        db                  = shelve.open(path_without_suffix, flag='r') # read only
        img_numpy_can       = db["image"]["canonical"]                   # 複数ステップ分が含まれている(1系列分)
        img_numpy_ran       = db["image"]["random_nonfix"]               # 複数ステップ分が含まれている(1系列分)
        # print("min: {} max: {}".format(img_numpy_can.min(), img_numpy_can.max())) # max=0, min=255
        # print("min: {} max: {}".format(img_numpy_ran.min(), img_numpy_ran.max())) # max=0, min=255
        # state = db["state"]
        # ctrl  = db["ctrl"]
        # import ipdb; ipdb.set_trace()
        step, width, height, channel = img_numpy_can.shape  # channlの順番に注意（保存形式に依存する）
        assert channel == 3

        img_torch_can = torch.zeros(step, channel, width, height)
        img_torch_ran = torch.zeros(step, channel, width, height)
        if self.transform is not None:
            for t in range(step):
                img_torch_can[t] = self.transform_all(img_numpy_can[t])
                img_torch_ran[t] = self.transform_all(img_numpy_ran[t])
        # print("min: {} max: {}".format(img_torch_can.min(), img_torch_ran.max())) # max=1.0, min=-1.0
        return index, img_torch_can, img_torch_ran


    def transform_all(self, img_numpy):
        '''
        - まずnumpy_arrayをPILとして読み込んでBGRをRGBに変換
        - ToTensor() が channel-first への入れ替と[0,1]への正規化をやってくれる
        - その後に[-1, 1]へと正規化する（デコーダの出力ネットワークの活性化関数がTanh()のため）
        '''
        x_min =  0.0; x_max = 1.0
        m     = -1.0;     M = 1.0

        img = np.flip(img_numpy, axis=-1).copy()
        img = self.transform(img)
        img = ((img - x_min) / (x_max - x_min)) * (M - m) + m
        return img


    def __len__(self):
        """ディレクトリ内の画像ファイルの数を返す。
        """
        return len(self.img_paths)

