import glob
import random
import os
import re

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_L(image):
    L_image = Image.new("L", image.size)
    L_image.paste(image)
    return L_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        self.path= os.path.join(root, mode)
        self.residual_path=os.path.join(root,"residual")
        self.num_dict={  '1': 14,
            '10': 12,
            '11': 10,
            '12': 9,
            '13': 11,
            '14': 8,
            '15': 11,
            '16': 13,
            '17': 10,
            '18': 11,
            '19': 12,
            '2': 14,
            '20': 13,
            '21': 12,
            '22': 11,
            '23': 11,
            '24': 10,
            '25': 13,
            '26': 12,
            '27': 12,
            '28': 11,
            '29': 14,
            '3': 12,
            '30': 15,
            '31': 14,
            '32': 12,
            '33': 13,
            '4': 12,
            '5': 13,
            '6': 13,
            '7': 13,
            '8': 15,
            '9': 13}

    def __getitem__(self, index):
        image_A_path=self.files[index % len(self.files)]
        image_A = Image.open(image_A_path)
        split_str=re.split("\D+",image_A_path)
        del split_str[0];del split_str[len(split_str)-1]

        image_B_name=split_str[0]+'_'+split_str[1]+'_'+str(int(split_str[2])+1)+'.png'
        image_C_name=split_str[0]+'_'+split_str[1]+'_'+str(int(split_str[2])+2)+'.png'
        if (int(split_str[2])+4) == self.num_dict[split_str[0]]:
            # print("if:split_str[2]=%d,num_dict[split_str[0]=%d" % (int(split_str[2]), self.num_dict[split_str[0]]))
            image_B_path=os.path.join(self.path,image_B_name)
            image_C_path=os.path.join(self.residual_path,image_C_name)
        elif (int(split_str[2])+3) == self.num_dict[split_str[0]]:
            # print("elif:split_str[2]=%d,num_dict[split_str[0]=%d" % (int(split_str[2]), self.num_dict[split_str[0]]))
            image_B_path=os.path.join(self.residual_path,image_B_name)
            image_C_path=os.path.join(self.residual_path,image_C_name)
        else:
            # print("else:split_str[2]=%d,num_dict[split_str[0]=%d" % (int(split_str[2]), self.num_dict[split_str[0]]))
            image_B_path=os.path.join(self.path,image_B_name)
            image_C_path=os.path.join(self.path,image_C_name)
        image_B = Image.open(image_B_path)
        image_C = Image.open(image_C_path)

        # Convert grayscale images to rgb
        if image_A.mode != "L":
            image_A = to_L(image_A)
        if image_B.mode != "L":
            image_B = to_L(image_B)
        if image_C.mode != "L":
            image_C = to_L(image_C)   

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        item_C = self.transform(image_C)
        return {"A": item_A, "B": item_B, "C": item_C}

    def __len__(self):
        return len(self.files)
