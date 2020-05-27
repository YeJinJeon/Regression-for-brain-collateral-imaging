import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt


class CenterCrop(object):
    """crop from center into 224"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    #class를 함수처럼 사용할 수 있도록
    def __call__(self, item):
        channels, depth, height, width = item.shape
        start_x = width // 2 - (self.output_size // 2)
        start_y = height // 2 - (self.output_size // 2)

        item = item[:, :, start_y:start_y + self.output_size, start_x:start_x + self.output_size]

        return item

class Fliplr(object):
    """Fliplr all items if prob == 1.0"""
    def __init__(self, prob):
        assert isinstance(prob, (float))
        self.prob = prob
    def __call__(self, item):
        item = item[:, :, :, ::-1]
        item = item.copy()
        return item

class BrainDataset(data.Dataset):
    def __init__(self, data_dir, data_idx):
        super(BrainDataset, self).__init__()
        self.data_dir = data_dir
        self.data_file = data_dir +'patient_list.xlsx'
        self.data_idx = data_idx #0:train, 1:val, 2: test
        self.df_idx = pd.read_excel(self.data_file, sheet_name = 'train_val_test_idx')
        self.mode_idx = self.df_idx.index[self.df_idx[0] == self.data_idx].to_list()
        self.ids = self.mode_idx

        self.transform = transforms.Compose([CenterCrop(224), Fliplr(1.0)])

    def view_dataset(self):
        img_file = "/media/data1/jeon/aug_result/IMG_n01.npy"
        img_cc_file = "/media/data1/jeon/aug_result/img_cc.npy"
        img_flip_file = "/media/data1/jeon/aug_result/img_flip.npy"

        img = np.load(img_file)
        print(img.shape)
        img_cc = np.load(img_cc_file)
        print(img_cc.shape)
        img_flip = np.load(img_flip_file)
        print(img_flip.shape)

        channels,depth,w,h=img.shape

        plt.imshow(img[30, 1, :, :])
        plt.show()
        plt.imshow(img_cc[30, 1, :, :])
        plt.show()
        plt.imshow(img_flip[30, 1, :, :])
        plt.show()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        #read filenames from excel
        df_filename = pd.read_excel(self.data_file, sheet_name='filenames')
        filepath = df_filename.loc[self.mode_idx, :].values.tolist()

        #modify filename
        if 'Workspace' in filepath[idx][0]:
            img_path = filepath[idx][0].replace("C:/Workspace/", self.data_dir)
        else:
            img_path = filepath[idx][0].replace("C:/workspace/", self.data_dir)

        #read img
        img = np.load(img_path) #IMG_n01.npy : already normalized (-0.9, 0.9)
        #print("img shape: ", img.shape) #(60, 20, 240, 240)

        # read ground truth
        gt_path_list = img_path.split('/')[0:-1]
        gt_path = '/'.join(gt_path_list) + '/phase_maps_medfilt_rs_n.npy'
        gt = np.load(gt_path)  # already normalized (0,1)
        #print("mask shape: ", mask.shape) #(5, 20, 240, 240)

        #read binary mask
        binary_mask_path = '/'.join(gt_path_list) + '/mask_4d.npy'
        binary_mask = np.load(binary_mask_path)
        #print("binary_mask shape: ", binary_mask.shape) #(1, 20, 240, 240)

        #read weight for loss
        weight_mask_path = '/'.join(gt_path_list) + '/phase_maps_wm.npy'
        weight_mask = np.load(weight_mask_path)
        #print("weight_mask shape: ", weight_mask.shape) #(5, 20, 240, 240)

        #threshold w/ binary mask -- already processed
        #img = img * binary_mask
        #gt = gt * binary_mask

        #augmentation
        img = self.transform(img)
        gt = self.transform(gt)
        binary_mask = self.transform(binary_mask)
        weight_mask = self.transform(weight_mask)

        return {'image': torch.from_numpy(img), 'gt': torch.from_numpy(gt), 'img_path': img_path, 'binary_mask': torch.from_numpy(binary_mask), 'weight_mask': torch.from_numpy(weight_mask)}


if __name__ == '__main__':

    data_dir = '/media/data1/jeon/workspace/'

    dataset = BrainDataset(data_dir, 0)


    #check dataloader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for batch in data_loader:
        imgpaths = batch['img_path']
        imgs = batch['image']
        masks = batch['mask']
        weight_masks = batch['weight_mask']
        print(imgpaths)
        print(imgs.shape)
        print(masks.shape)
        print(weight_masks.shape)

    """
    #check augmentation result
    filepath = "/media/data1/jeon/aug_result/IMG_n01.npy"
    centercrop = CenterCrop(224)
    fliplr = Fliplr(1.0)

    #save augmented image
    img = np.load(filepath)
    img_cc = centercrop(img)
    print(img_cc.shape)
    np.save("/media/data1/jeon/aug_result/img_cc.npy", img_cc)
    img_flip = fliplr(img_cc)
    print(img_flip.shape)
    np.save("/media/data1/jeon/aug_result/img_flip.npy", img_flip)

    #veiw augmented image
    dataset.view_dataset()

    print("finish saving augmentation images")
    """


