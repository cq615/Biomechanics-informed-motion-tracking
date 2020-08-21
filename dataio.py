import torch.utils.data as data
from os import listdir
from os.path import join
import numpy as np
import nibabel as nib
import glob
import cv2
from scipy import ndimage
from util import *


class TrainDataset_motion(data.Dataset):
    """Dataloader for loading biomechanically simulated motion fields"""
    def __init__(self, data_path, split_set, img_size=96):
        super(TrainDataset_motion, self).__init__()
        self.data_path = data_path
        self.split_set = split_set
        self.img_size = img_size
        filename = [f.split('_')[0] for f in sorted(listdir(join(self.data_path, self.split_set)))]
        self.filename = list(set(filename))

    def __getitem__(self, index):
        disp, mask = load_motion_df(join(self.data_path, self.split_set), self.filename[index], self.img_size)

        return disp, mask

    def __len__(self):
        return len(self.filename)


def load_motion_df(data_path, filename, img_size, rand_frame=None):
    # Load motion deformation fields and segmentation maps
    s_num = [f.split('_')[2] for f in glob.glob(join(data_path, filename+'_slice_*_disp.npy'))]
    slice_n = np.random.choice(s_num)

    disp = np.load(join(data_path, filename+'_slice_'+slice_n+'_disp.npy'))
    mask = np.load(join(data_path, filename+'_slice_'+slice_n+'_ED.npy'))

    # generate random index for t and z dimension
    if rand_frame is not None:
        rand_t = rand_frame
    else:
        rand_t = np.random.randint(0, disp.shape[0])

    # dilate myocardial mask to include surrounding context
    mask = mask.astype(np.int16)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)

    disp = disp[rand_t]
    disp = centre_crop(disp, size=img_size, centre=[disp.shape[1]//2, disp.shape[2]//2])
    mask = centre_crop(mask[np.newaxis], size=img_size, centre=[disp.shape[1] // 2, disp.shape[2] // 2])

    disp = disp / (disp.shape[1] // 2)  # normalise deformation to [-1, 1]
    mask = np.transpose(mask, (0, 2, 1))
    disp = np.transpose(disp, (0, 2, 1))
    disp = np.array(disp, dtype='float32')
    mask = np.array(mask, dtype='int16')

    return disp, mask


class TrainDataset(data.Dataset):
    def __init__(self, data_path):
        super(TrainDataset, self).__init__()
        self.data_path = data_path
        self.filename = [f for f in sorted(listdir(self.data_path))]

    def __getitem__(self, index):
        input, target, mask = load_data(self.data_path, self.filename[index], size=96)

        image = input[:1]
        image_pred = input[1:]

        return image, image_pred, target, mask

    def __len__(self):
        return len(self.filename)


class ValDataset(data.Dataset):
    def __init__(self, data_path):
        super(ValDataset, self).__init__()
        self.data_path = data_path
        self.filename = [f for f in sorted(listdir(self.data_path))]

    def __getitem__(self, index):
        input, target, mask = load_data(self.data_path, self.filename[index], size=96, rand_frame=index % 30)

        image = input[:1]
        image_pred = input[1:]

        return image, image_pred, target, mask

    def __len__(self):
        return len(self.filename)


def load_data(data_path, filename, size, rand_frame=None):
    # Load images and labels
    nim = nib.load(join(data_path, filename, 'sa.nii.gz'))
    image = nim.get_data()[:, :, :, :]
    image = np.array(image, dtype='float32')

    # generate random index for t and z dimension
    if rand_frame is not None:
        rand_t = rand_frame
        rand_z = rand_frame % (image.shape[2]-1)+1
    else:
        rand_t = np.random.randint(0, image.shape[3])
        rand_z = np.random.randint(1, image.shape[2]-1)

    # preprocessing
    image_max = np.max(np.abs(image))
    image /= image_max
    image_sa = image[..., rand_z, rand_t]
    image_sa = image_sa[np.newaxis]

    nim = nib.load(join(data_path, filename, 'sa_'+'ED'+'.nii.gz'))
    image = nim.get_data()[:, :, :]
    image = np.array(image, dtype='float32')

    nim_seg = nib.load(join(data_path, filename, 'label_sa_'+'ED'+'.nii.gz'))
    seg = nim_seg.get_data()[:, :, :]

    image_ED = image[..., rand_z]
    image_ED /= image_max
    seg_ED = seg[..., rand_z]

    slice = (seg[..., image.shape[2]//2] == 2).astype(np.uint8)
    centre = ndimage.measurements.center_of_mass(slice)
    centre = np.round(centre).astype(np.uint8)

    image_ED = image_ED[np.newaxis]
    seg_ED = seg_ED[np.newaxis]

    image_bank = np.concatenate((image_sa, image_ED), axis=0)

    image_bank = centre_crop(image_bank, size, centre)
    seg_ED = centre_crop(seg_ED, size, centre)
    image_bank = np.transpose(image_bank, (0, 2, 1))
    seg_ED = np.transpose(seg_ED, (0, 2, 1))
    image_bank = np.array(image_bank, dtype='float32')
    seg_ED = np.array(seg_ED, dtype='int16')

    mask = (seg_ED == 2).astype(np.uint8)
    mask = centre_crop(mask, size, centre)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask[0], kernel, iterations=3)
    mask = np.array(mask[np.newaxis], dtype='int16')

    return image_bank, seg_ED, mask
