from __future__ import print_function, division
import os
from skimage import io
from torch.utils.data import Dataset
import glob

class MyDataSet(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.__len__


    def listdirs(self,path):
        seen = set()
        for root, dirs, files in os.walk(path, topdown=False):
            if dirs:
                parent = root
                while parent:
                    seen.add(parent)
                    parent = os.path.dirname(parent)
            for d in dirs:
                d = os.path.join(root, d)
                if d not in seen:
                    yield d

    def __len__(self):
        dirs = self.listdirs(self.root_dir)
        all = []
        for d in dirs:
         print (d)
         all.append(glob.glob(d.replace("\\","/")+'/*.jpg'))
        print(all)
        flat_list = []
        for sublist in all:
            for item in sublist:
                flat_list.append(item)
        self.all = flat_list
        print (len (self.all))
        return len (self.all)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.all[idx])
        img_name = img_name.replace("\\","/")
        print (img_name)
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
