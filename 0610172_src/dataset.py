
from skimage import io
import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import torch


class thumbnail128(Dataset):
    def __init__(self, root_dir, img_path, transform):
        ##############################################
        ### Initialize paths, transforms
        ##############################################
        
        # load image path
        self.img_path = pd.read_csv(img_path,sep = "	",header = None)
        self.img_path = self.img_path.values
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):        
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.img_path)

    def __getitem__(self, idx):
        ##############################################
        # 1. Read from file
        # 2. Preprocess the data 
        # 3. Return the data
        ##############################################
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir,self.img_path[idx][0])
        image = Image.open(img_name)
        image = self.transform(image)
        
        return image
