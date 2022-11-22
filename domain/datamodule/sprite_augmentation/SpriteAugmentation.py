import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Callable


class SpriteAugmentation(VisionDataset):
    ''' Sprite Dataset
        - sequence
            - train: 6687
            - test : 873
        - step              : 8
        - image size        : (3, 64, 64)
        - action variation  : 9
            - 歩いたり，手を振ったりなど
        - minmax value:
            - min: -1.0
            - max:  1.0
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
        self.min       =  -1.0
        self.max       =   1.0


    def _get_img_paths(self, img_dir):
        """
        指定したディレクトリ内の画像ファイルのパス一覧を取得する。
        """
        if self.train: img_dir = img_dir + "/Sprite/lpc-dataset/train/"
        else         : img_dir = img_dir + "/Sprite/lpc-dataset/test"
        img_dir = Path(img_dir)
        img_paths = [p for p in img_dir.iterdir() if p.suffix == ".sprite"]
        # import ipdb; ipdb.set_trace()
        img_paths = self._paths_sorted(img_paths)
        return img_paths


    def _paths_sorted(self, paths):
        '''
        ・x.stem はそのままの意味でxが持つstemという変数にアクセスしている
        ・ここでxはpathsの要素になるのでPosixPathに対応する
        ・従って，やっていることは PosixPath(~).stem と等価
        ・PosixPath(~).stem の中には整数値 int が string の形で格納されている
        '''
        return sorted(paths, key = lambda x: int(x.stem))


    def __len__(self):
        """ディレクトリ内の画像ファイルの数を返す。
        """
        return len(self.img_paths)


    def __getitem__(self, index: int):
        path = self.img_paths[index] # Ex.) PosixPath('data/Sprite/lpc-dataset/train/1808.sprite')
        img  = torch.load(str(path)) # img.shape = torch.Size([8, 3, 64, 64]) になるので１つのパスからロードしたデータに複数ステップ分が含まれている
        # step, channel, width, height = img.shape

        img_aug_context  = self.augment_context(img)
        img_aug_dynamics = self.augment_dynamics(img)
        return index, img, img_aug_context, img_aug_dynamics


    def augment_context(self, img):
        '''
        The static factor (e.g., the character identity in videos or the speaker in audios) is shared
        across all the time steps, and should not be affected by the exact order of the frames.
        We therefore randomly shuffle or simply reverse the order of time steps to generate the content augmentation of
        '''
        step         = img.shape[0]
        random_index = torch.randperm(step)
        return torch.index_select(img, dim=0, index=random_index)


    def augment_dynamics(self, img):
        '''
        combination of
            - (cropping)
            - color distortion
            - Gaussian blur
            - reshaping
        '''
        img = self.color_distortion(img)
        # img = self.gaussian_blur(img)
        return img


    def color_distortion(self, img):
        img = (img - self.min) / (self.max - self.min) # scale to [0, 1]
        img = torch.Tensor([1.0]) - img                # inverse color (= 255 - original_RGB)
        return img


if __name__ == '__main__':
    import numpy as np
    import cv2
    from torchvision import transforms

    loader  = SpriteAugmentation("data/Sprite/", train=False, transform=transforms.ToTensor())

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    for i,dataitem in enumerate(loader):
        # _,_,_,_,_,_,data = dataitem
        print("loader : ", i)

        images = []
        for k, d in enumerate(dataitem):
            # print("dataitem : ", k)
            d = np.array(d) # (step, channel, w, h)
            dt = np.transpose(d, (1, 2, 0))
            cv2.imshow("img", dt)
            cv2.waitKey(50)

