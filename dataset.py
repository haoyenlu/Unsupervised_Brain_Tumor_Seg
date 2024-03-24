from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import h5py
from torchvision.transforms import ToTensor, Normalize, Resize
import glob
import os

class TumorImageDataset(Dataset):
  def __init__(self,root,size=(512,512),is_train=True,split_ratio = 0.8):
    super().__init__()
    self.root = root
    self.is_train = is_train
    self.split_ratio = split_ratio
    self.size = size

    all_files = [os.path.basename(x) for x in glob.glob(os.path.join(self.root,"*.mat"))]
    all_files.remove('cvind.mat')
    train_split = np.ceil(self.split_ratio * len(all_files)).astype(int)
    
    if is_train:
      self.files = all_files[:train_split]
    else:
      self.files = all_files[train_split:]


  def __len__(self):
    return len(self.files)

  def __getitem__(self,index):
    if torch.is_tensor(index): index.tolist()

    image_path = self.files[index]
    image_file = h5py.File(os.path.join(self.root,image_path))

    image = image_file["cjdata"]["image"][:,:].astype(np.float32)
    mask = image_file["cjdata"]["tumorMask"][:,:]
    label = image_file["cjdata"]["label"][:,:][0].astype(int)

    image = self.transform_image(image)
    mask = self.transform_mask(mask)
    
    sample = {"image":image,"mask":mask,"label":label}
    return sample

  def transform_image(self,image):
    image_stack = np.stack((image,image,image),axis=2)
    image_tensor = ToTensor()(image_stack)
    min_val, max_val = torch.min(image_tensor), torch.max(image_tensor)
    image_normal = (image_tensor - min_val) / (max_val - min_val)
    image_resize = Resize(self.size)(image_normal)
    return image_resize

  def transform_mask(self,mask):
    mask_tensor = ToTensor()(mask)
    mask_resize = Resize(self.size)(mask_tensor)
    mask_squeeze = torch.squeeze(mask_resize)
    return mask_squeeze
