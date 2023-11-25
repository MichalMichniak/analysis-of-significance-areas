from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import OneHotEncoder
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection
from Bigearth import Bigearth,Dataloader,Bigearth_Pruned
from typing import Tuple
from tqdm import tqdm

def generate_new_inst(img_num : int,last_idx_empty : int, dataset : Dataset, path : str, ) -> Tuple[int, np.ndarray]:
    list_transformations = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    no_transform = np.random.randint(0,3)
    list_transformations.pop(no_transform)
    X,y = dataset[img_num]
    I = cv2.rotate(X, list_transformations[0])
    I = np.clip(((I.astype(float))*255), 0, 255).astype("uint8")
    cv2.imwrite(f"{path}\{last_idx_empty}.png", cv2.cvtColor(I, cv2.COLOR_RGB2BGR))
    idx1 = last_idx_empty
    last_idx_empty += 1
    I = cv2.rotate(X, list_transformations[1])
    I = np.clip(((I.astype(float))*255), 0, 255).astype("uint8")
    cv2.imwrite(f"{path}\{last_idx_empty}.png", cv2.cvtColor(I, cv2.COLOR_RGB2BGR))
    idx2 = last_idx_empty
    last_idx_empty += 1
    return last_idx_empty, np.array([np.insert(y,0,idx1),np.insert(y,0,idx2)])

def test_train_split_generate_to_csv(dataset : Dataset, multiplicate : np.ndarray, path : str, test_size=0.2, random_seed = 23421):
    indexes = pd.read_csv("./labels/labels_cut.csv")
    train , test = sklearn.model_selection.train_test_split(indexes, test_size=test_size, random_state=random_seed)
    last_idx_empty = dataset.get_start_generated()
    for img_num, tqdm_ in zip(train["idx"],tqdm(range(1,train["idx"].__len__()+1))):
        y = dataset.get_y(img_num)
        if(sum(y * (multiplicate - 1))>0):
            last_idx_empty, train_append = generate_new_inst(img_num, last_idx_empty, dataset, path)
            train = train.append(pd.DataFrame(train_append, columns=train.columns), ignore_index=True)
    train.to_csv("./labels/train_labels.csv", index=False)
    test.to_csv("./labels/test_labels.csv", index=False)


if __name__ == '__main__':
    ds = Bigearth_Pruned()
    test_train_split_generate_to_csv(ds, np.array([3,1,1,1,1,1,1,1,1,3,1,1,1]), "C:\D\VS_programs_python\inzynierka\BigearthNet_png_gen")