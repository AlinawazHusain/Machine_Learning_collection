from PIL import Image , ImageFile
import numpy as np 
from torch.utils.data import Dataset
import os
import config

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Custdata(Dataset):
    def __init__(self ,root_dir,):

        self.root = root_dir
        self.list_files = os.listdir(self.root)

    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self , index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root , img_file)

        img = np.array(Image.open(img_path))
        width = img.shape[1]

        input_img = img[: , :width//2 , :]
        target_img = img[: width//2: , :]


        augmentations = config.transform_both(image = input_img , image0 = target_img)
        input_img = augmentations['image']
        target_img = augmentations['image0']

        input_img = config.transform_only_input(image = input_img)['image']
        target_img = config.transform_only_mask(target_img)['image']

        return input_img , target_img 
    
    
        