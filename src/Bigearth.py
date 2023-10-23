from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import OneHotEncoder
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection
import torchvision.io

# class Bigearth(Dataset):
#     def __init__(self):
#         # self.X = torch.from_numpy(np.array(x_train)).type(torch.float)
#         # enc = OneHotEncoder(handle_unknown='ignore')
#         # enc = enc.fit(y_train)
#         self.y = pd.read_csv("C:\D\VS_programs_python\inzynierka\\analysis-of-significance-areas\\labels.csv")
#         self.database_len = 590_326#500
#         self.start_generated = 590_327
#         self.n_samples = self.database_len
    
#     def __getitem__(self, index):
#         self.X = plt.imread(f"C:\D\VS_programs_python\inzynierka\BigearthNet_png\{index}.png")
#         return self.X, self.y.iloc[index-1].values[1:]
    
#     def __len__(self):
#         return self.n_samples
    
#     def get_start_generated(self):
#         return self.start_generated

class Bigearth_Pruned(Dataset):
    def __init__(self):
        # self.X = torch.from_numpy(np.array(x_train)).type(torch.float)
        # enc = OneHotEncoder(handle_unknown='ignore')
        # enc = enc.fit(y_train)
        self.y = pd.read_csv("C:\D\VS_programs_python\inzynierka\\analysis-of-significance-areas\\labels\\labels_idx.csv")
        self.database_len = len(self.y)#500#
        self.database_len_extended = 676_856#700
        self.start_generated = 590_327
        self.n_samples = self.database_len_extended
    
    def __getitem__(self, index):
        self.X = torchvision.io.read_image(f"C:\D\VS_programs_python\inzynierka\BigearthNet_png\{self.y.iloc[index-1].values[0]}.png")
        self.X = self.X.float().cuda()
        return self.X, self.y.iloc[index-1].values[1:]
    
    def get_y(self, index):
        return self.y.iloc[index-1].values[1:]

    def __len__(self):
        return self.n_samples
    
    def get_start_generated(self):
        return self.start_generated


class Train_Dataset(Dataset):
    def init_classes(self):
        self.class_groups = []
        for col in self.train.columns[1:]:
            self.class_groups.append(self.train[self.train[col] == 1])

    def __init__(self, dataset : Dataset, test_size=0.2, random_seed = 23421, batch_len = 1, class_instances = 50_000):
        self.train = pd.read_csv("C:\D\VS_programs_python\inzynierka\\analysis-of-significance-areas\\labels\\train_labels.csv")
        # self.test = pd.read_csv("C:\D\VS_programs_python\inzynierka\\analysis-of-significance-areas\\test_labels.csv")
        self.dataset_ = dataset
        # self.batch_len = batch_len
        self.class_instances = class_instances
        self.epoch_data = 0
        # self.data_len = class_instances*len(self.train.columns[1:])//batch_len
        self.number_of_instances_epoch = class_instances*len(self.train.columns[1:])
        self.init_classes()
        self.next_epoch_train_set()
    
    def next_epoch_train_set(self):
        """
        funnction to resample instances of a classes
        """
        samples = []
        for group in self.class_groups:
            samples.append(group.sample(self.class_instances))
        samples = pd.concat(samples)
        self.epoch_data = samples.sample(frac = 1)

    def __getitem__(self, index):
        x,y = self.dataset_[self.epoch_data.iloc[index]["idx"]]
        return x,y
        
    
    def __len__(self):
        return len(self.epoch_data)



class Test_Dataset(Dataset):
    # def init_classes(self):
    #     self.class_groups = []
    #     for col in self.train.columns[1:]:
    #         self.class_groups.append(self.train[self.train[col] == 1])

    def __init__(self, dataset : Dataset, test_size=0.2, random_seed = 23421, batch_len = 1, class_instances = 50_000):
        # self.train = pd.read_csv("C:\D\VS_programs_python\inzynierka\\analysis-of-significance-areas\\train_labels.csv")
        self.test = pd.read_csv("C:\D\VS_programs_python\inzynierka\\analysis-of-significance-areas\\labels\\test_labels.csv")
        self.dataset_ = dataset
        # self.batch_len = batch_len
        self.class_instances = class_instances
        # self.epoch_data = 0
        # self.data_len = class_instances*len(self.train.columns[1:])//batch_len
        self.number_of_instances_epoch = class_instances*len(self.test.columns[1:])
        # self.init_classes()
        # self.next_epoch_train_set()
    
    # def next_epoch_train_set(self):
    #     """
    #     funnction to resample instances of a classes
    #     """
    #     samples = []
    #     for group in self.class_groups:
    #         samples.append(group.sample(self.class_instances))
    #     samples = pd.concat(samples)
    #     self.epoch_data = samples.sample(frac = 1)

    def __getitem__(self, index):
        x,y = self.dataset_[self.test.iloc[index]["idx"]]
        return x,y
        
    
    def __len__(self):
        return len(self.test)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    bigearth_dl = Bigearth_Pruned()
    dataset = Train_Dataset(bigearth_dl,batch_len=8)
    dataloader = DataLoader(dataset=dataset, batch_size=8,shuffle= True,num_workers=4 )
    print(dataloader.__len__())
    x,y = next(iter(dataloader))
    for im in x:
        print(im)
        plt.imshow(im.T)
        plt.show()
    data_iter = iter(dataloader)
    for i in range(100):
        x,y = next(data_iter)
        print(i)
    for i in range(100):
        x,y = dataset[i]
        print(i)
    print(y)