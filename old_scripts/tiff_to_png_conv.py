import cv2
import numpy as np
import matplotlib.pyplot as plt

def tiff_pach_to_png(n,path_input,path_output):
    lst = [4,3,2]
    I = []
    for i in lst:
        I .append(plt.imread(f"..\BigEarthNet-S2\BigEarthNet-v1.0\{path_input}\{path_input}_B0{i}.tif")) 
    I = np.array(I).T
    I = cv2.resize(I,(224,224))
    I = np.clip(((I.astype(float))*255)/(2000.0), 0, 255).astype("uint8")
    cv2.imwrite(f"{path_output}\{n}.png",I[:,:,::-1])


if __name__ == '__main__':
    import os
    import re
    from tqdm import tqdm
    #print(os.walk("..\BigEarthNet-S2\BigEarthNet-v1.0"))
    database_len = 590_326
    i = database_len + 2
    k = 590_326
    enum_dir = enumerate(os.walk("..\BigEarthNet-S2\\BigEarthNet-v1.0"))
    for i in zip(range(k),tqdm(range(k))):
        next(enum_dir)
    for x,tqdm_temp in zip(enum_dir,tqdm(range(k,database_len+1))):
        n,x = x
        name = re.search(r'S2(A|B).*',x[0])
        if name != None:
            #print(n,name.group())
            tiff_pach_to_png(n,name.group(),"..\BigearthNet_png")
        if n == i:
            break
    print("Finished")