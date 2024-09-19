from cProfile import label
from email.mime import image
from re import M
from torchvision.transforms import transforms
import os
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torch
import random
import numpy as np
import torch
from torch import nn
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
np.random.seed(0)


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

def expand_greyscale(t):
    return t.expand(3, -1, -1)

def read_split_data_T(root:str, ratio=0):
    dataset_path = []
    val= []
    if True:
        class_list = [c for c in  os.listdir(root) if os.path.isdir(os.path.join(root,c))]
        class_list.sort()
        for c in class_list: 
                cla_path = os.path.join(root,c)
                images = [os.path.join(root, c,i) for i in os.listdir(cla_path)]
                for img_path  in images:
                    dataset_path.append(img_path)
    return dataset_path,val

def center(path,resolution_r):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((resolution_r,resolution_r)),
                                    transforms.ToTensor()])
    img = transform(img)
    return img

def center_Aug(path,resolution_r):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((resolution_r,resolution_r)),
                                    transforms.ToTensor(),
                                    transforms.RandomRotation(35),
                                    RandomApply( transforms.ColorJitter(brightness=0,contrast=0, saturation=0, hue=0.25),p=0.2),
                                    transforms.RandomGrayscale(p=0.2),
                                    transforms.RandomHorizontalFlip(),
                                    RandomApply(transforms.GaussianBlur((3, 3), (1.0, 2.0)), p = 0.4),
                                    transforms.RandomResizedCrop(size=(resolution_r,resolution_r ),scale=(0.75,1)),
                                    ])
    img = transform(img)
    return img

def center_TarTest(path,resolution_r):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((resolution_r,resolution_r)),
                                    transforms.ToTensor(),])
    img = transform(img)
    return img

    
class GaussianBlur(object):
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )
        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)
        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))
        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()
        img = self.tensor_to_pil(img)
        return img


class SourceDataset(Dataset):
    def __init__(self,path,root,resolution,tarpath):        
        self.path = path
        self.root = root
        self.tarpath = tarpath
        self.resolution = resolution
        self.domain_list = [c for c in os.listdir(self.root) if os.path.isdir(os.path.join(self.root,c))]
        self.domain_list.sort()
        self.Msdomain = len(self.domain_list)
        root1 = os.path.join(self.root,self.domain_list[0])
        self.class_list = [c for c in  os.listdir(root1) if os.path.isdir(os.path.join(root1,c))]
        self.class_list.sort()

    def __len__(self):
        return len(self.path)
        
    def __getitem__(self, index): 
        image_path = os.path.join(self.path[index])
        train_domain = image_path.split('/')[-3]
        train_label = image_path.split('/')[-2]
        pos_domain = random.randint(0,self.Msdomain-1)
        while self.domain_list[pos_domain] == train_domain and self.Msdomain > 1:
            pos_domain = random.randint(0,self.Msdomain-1)
        path_temp = os.path.join(self.root,self.domain_list[pos_domain],train_label)
        pos_images = [os.path.join(self.root,self.domain_list[pos_domain],train_label,i)for i in os.listdir(path_temp)]

        img_pos_path = random.sample(pos_images, k=1)[0]
        image_anchor = center_TarTest(image_path,resolution_r=self.resolution)
        image_positive = center_TarTest(img_pos_path,resolution_r=self.resolution)

        train_domain = self.domain_list.index(train_domain)
        train_label = self.class_list.index(train_label)

        return image_anchor,image_positive,train_label,train_domain,pos_domain

class TargetDataset(Dataset):
    def __init__(self,tarpath,args,resolution,root,target):  
        self.tarpath = tarpath
        self.root = root
        self.resolution = resolution
        self.tar = target
        root1 = os.path.join(self.root,self.tar)
        self.class_list = [c for c in  os.listdir(root1) if os.path.isdir(os.path.join(root1,c))]
        self.class_list.sort()
        
    def __len__(self):
        return len(self.tarpath)

    def __getitem__(self, index): 
        
        tar_path = os.path.join(self.tarpath[index])
        tar_label = tar_path.split('/')[-2]
        tar_image = center_TarTest(tar_path,resolution_r=self.resolution)
        tar_label = self.class_list.index(tar_label)
        return tar_image,tar_label
