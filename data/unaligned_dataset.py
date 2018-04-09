
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'

from random import shuffle, seed
# seed(1234)
class UnalignedDatasetHalf(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        directory = os.path.join(opt.dataroot, opt.phase)
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        classes = os.listdir(directory)
        classes.sort()
        self.data = []
        for i, c in enumerate(classes):
            images = os.listdir(os.path.join(directory, c))
            images = [os.path.join(directory, c, img) for img in images if '.jpg' in img]
            self.data.append([{'image': img, 'label': i} for img in images])

        if opt.n_classes!=None:
            assert(opt.n_classes == len(classes))
        else:
            opt.n_classes = len(classes)

        # 

        # self.A_paths = make_dataset(self.dir_A)
        # self.B_paths = make_dataset(self.dir_B)

        # self.A_paths = sorted(self.A_paths)
        # self.B_paths = sorted(self.B_paths)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        # TODO randomize class selection
        class_A = 0 #random.randint(0,len(self.data)-1)
        class_B = random.randint(0,len(self.data)-1)
        while class_B==class_A:
            class_B = random.randint(0,len(self.data)-1)

        self.A_size = len(self.data[class_A])
        self.B_size = len(self.data[class_B])
        index_A = index
        A_path = self.data[class_A][index_A % self.A_size]['image']
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.data[class_B][index_B]['image']
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_image = self.transform(A_img)
        B_image = self.transform(B_img)
        # if self.opt.which_direction == 'BtoA':
        #     input_nc = self.opt.output_nc
        #     output_nc = self.opt.input_nc
        # else:
        #     input_nc = self.opt.input_nc
        #     output_nc = self.opt.output_nc

        # if input_nc == 1:  # RGB to gray
        #     tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        #     A = tmp.unsqueeze(0)

        # if output_nc == 1:  # RGB to gray
        #     tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
        #     B = tmp.unsqueeze(0)

        A_label = self.data[class_A][index_A % self.A_size]['label']
        A_label = torch.LongTensor([A_label])
        B_label = self.data[class_B][index_B % self.B_size]['label']
        B_label = torch.LongTensor([B_label])

        return {'A': A_image, 'B': B_image,
                'A_label': A_label, 'B_label':B_label,
                'A_paths': A_path , 'B_paths': B_path
                }

    def __len__(self):
        return max([len(arr) for arr in self.data])
    
    def n_classes(self):
        return len(self.data)

    def name(self):
        return 'UnalignedDatasetHalf'