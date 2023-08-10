import os, torch
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL

class MF_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=480, input_w=640, transform=[], length=None, is_supervised=True):
        super(MF_dataset, self).__init__()

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted', 'train_sup', 'train_unsup'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.length = length
        self.n_data    = len(self.names)
        self.is_sup = is_supervised

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image     = np.asarray(PIL.Image.open(file_path)).copy()
        return image
    
    def __len__(self):
        if self.length is not None:
            return self.length
        return self.n_data
    
    def _construct_new_file_names(self, length):
        assert isinstance(length, int)

        if length < self.n_data:
            return self.names[:length]

        new_names = self.names * (length // self.n_data)

        rand_indices = torch.randperm(self.n_data).tolist()
        new_indices = rand_indices[:length % self.n_data]

        new_names += [self.names[i] for i in new_indices]

        return new_names

    def __getitem__(self, index):
        if self.length is not None:
            name = self._construct_new_file_names(self.length)[index]
        else:
            name = self.names[index]        

        image = self.read_image(name, 'images')
        label = self.read_image(name, 'labels')
            
        if self.is_sup:
            for func in self.transform:
                image, label = func(image, label)

            image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)))
            image = image.astype('float32')
            image = np.transpose(image, (2,0,1))/255.0
            label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST))
            label = label.astype('int64')
            
            return torch.tensor(image), torch.tensor(label), name
        else:
            for func in self.transform[:-1]:
                image, label = func(image, label)
            
            image_mco, _ = self.transform[-1](image.copy(), label)
                
            image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)))
            image = image.astype('float32')
            image = np.transpose(image, (2,0,1))/255.0
            
            image_mco = np.asarray(PIL.Image.fromarray(image_mco).resize((self.input_w, self.input_h)))
            image_mco = image_mco.astype('float32')
            image_mco = np.transpose(image_mco, (2,0,1))/255.0
            
            return torch.tensor(image), torch.tensor(image_mco)

