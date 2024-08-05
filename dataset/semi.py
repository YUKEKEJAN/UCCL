# from dataset.transform import *

# from copy import deepcopy
# import math
# import numpy as np
# import os
# import random

# from PIL import Image
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms



# class SemiDataset_val(Dataset):
#     def __init__(self,  root,  size=None, id_path=None, nsample=None):

#         self.root = root


#         self.ids = os.listdir(root)

#     def __getitem__(self, item):
#         id = self.ids[item]
#         img = Image.open(os.path.join(self.root, id)).convert('RGB')
#         img = crop_single(img, 801, 255)
#         image_pp = "/home/user/Original/"
        
#         img.save(image_pp + str(id))
#         img = normalize(img)
#         return img,id
    
#     def __len__(self):
#         return len(self.ids)
        
# import copy
# class SemiDataset(Dataset):
#     def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
#         self.name = name
#         self.root = root
#         self.mode = mode
#         self.size = size

#         if mode == 'train_l' or mode == 'train_u':
#             with open(id_path, 'r') as f:
#                 self.ids = f.read().splitlines()
#             if mode == 'train_l' and nsample is not None:
#                 self.ids *= math.ceil(nsample / len(self.ids))
#                 self.ids = self.ids[:nsample]
#         else:
#             with open('splits/%s/val.txt' % name, 'r') as f:
#                 self.ids = f.read().splitlines()

#     def __getitem__(self, item):
#         id = self.ids[item]
#         img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
#         mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))
#         raw_image = copy.deepcopy(img).convert('RGB')
#         raw_image = np.array(raw_image)
#         raw_image = torch.from_numpy(raw_image)
#         if self.mode == 'val':
#             img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)
#             if random.random() < 0.8:
#                 img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
#             img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
#             img_s1 = blur(img_s1, p=0.5)
#             img_w, mask = normalize(img_w,mask)
#             img_s1 = normalize(img_s1)
#             return raw_image,img_w,img_s1,mask
#             # img, mask = normalize(img, mask)
#             # return raw_image,img, mask, id

#         img, mask = resize(img, mask, (0.5, 2.0))
#         ignore_value = 254 if self.mode == 'train_u' else 255
#         img, mask = crop(img, mask, self.size, ignore_value)
#         img, mask = hflip(img, mask, p=0.5)

#         if self.mode == 'train_l':
#             return normalize(img, mask)

#         img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

#         if random.random() < 0.8:
#             img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
#         img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
#         img_s1 = blur(img_s1, p=0.5)
#         cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

#         if random.random() < 0.8:
#             img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
#         img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
#         img_s2 = blur(img_s2, p=0.5)
#         cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

#         ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

#         img_s1, ignore_mask = normalize(img_s1, ignore_mask)
#         img_s2 = normalize(img_s2)

#         mask = torch.from_numpy(np.array(mask)).long()
#         ignore_mask[mask == 254] = 255

#         # 返回的是弱增强图片的正则化，  两张强增强图片， 
#         return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

#     def __len__(self):
#         return len(self.ids)

from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
    


class SemiDataset_2(Dataset):
    def __init__(self, root):

        self.root = root

        self.ids = os.listdir(self.root)

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id)).convert('RGB')
        # mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        img = normalize(img)
        return img , id
    
    def __len__(self):
        return len(self.ids)
