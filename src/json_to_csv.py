import json
import os
import re
from tqdm import tqdm

LABELS_DS = ['Agro-forestry areas', 'Airports', 'Annual crops associated with permanent crops', 'Bare rock',
 'Beaches dunes sands', 'Broad-leaved forest', 'Burnt areas', 'Coastal lagoons', 'Complex cultivation patterns',
 'Coniferous forest', 'Construction sites', 'Continuous urban fabric', 'Discontinuous urban fabric', 'Dump sites',
 'Estuaries', 'Fruit trees and berry plantations', 'Green urban areas', 'Industrial or commercial units',
 'Inland marshes', 'Intertidal flats', 'Land principally occupied by agriculture with significant areas of natural vegetation',
 'Mineral extraction sites', 'Mixed forest', 'Moors and heathland', 'Natural grassland', 'Non-irrigated arable land',
 'Olive groves', 'Pastures', 'Peatbogs', 'Permanently irrigated land', 'Port areas', 'Rice fields',
 'Road and rail networks and associated land', 'Salines', 'Salt marshes', 'Sclerophyllous vegetation', 'Sea and ocean',
 'Sparsely vegetated areas', 'Sport and leisure facilities', 'Transitional woodland/shrub', 'Vineyards', 'Water bodies',
 'Water courses']

def json_to_csv():
    
    labels = ['Agro-forestry areas', 'Airports', 'Annual crops associated with permanent crops', 'Bare rock',
 'Beaches, dunes, sands', 'Broad-leaved forest', 'Burnt areas', 'Coastal lagoons', 'Complex cultivation patterns',
 'Coniferous forest', 'Construction sites', 'Continuous urban fabric', 'Discontinuous urban fabric', 'Dump sites',
 'Estuaries', 'Fruit trees and berry plantations', 'Green urban areas', 'Industrial or commercial units',
 'Inland marshes', 'Intertidal flats', 'Land principally occupied by agriculture, with significant areas of natural vegetation',
 'Mineral extraction sites', 'Mixed forest', 'Moors and heathland', 'Natural grassland', 'Non-irrigated arable land',
 'Olive groves', 'Pastures', 'Peatbogs', 'Permanently irrigated land', 'Port areas', 'Rice fields',
 'Road and rail networks and associated land', 'Salines', 'Salt marshes', 'Sclerophyllous vegetation', 'Sea and ocean',
 'Sparsely vegetated areas', 'Sport and leisure facilities', 'Transitional woodland/shrub', 'Vineyards', 'Water bodies',
 'Water courses']
    
    dictionary_labels = {i:n for n,i in enumerate(labels)}
    database_len = 590_326
    print(len(labels))
    i = database_len
    with open("./labels.csv","w") as file:
        file.write(str(["idx"]+LABELS_DS)[1:-1]+"\n")
        for x,tqdm_temp in zip(enumerate(os.walk("..\BigEarthNet-S2\\BigEarthNet-v1.0")),tqdm(range(database_len))):
            n,x = x
            name = re.search(r'S2(A|B).*',x[0])
            if name != None:
                f = open(f"..\BigEarthNet-S2\BigEarthNet-v1.0\{name.group()}\{name.group()}_labels_metadata.json",)
                data = json.load(f)["labels"]
                one_hot_encoding = [0 for k in range(len(labels))]
                for type in data:
                    one_hot_encoding[dictionary_labels[type]] = 1
                #### get all types
                # for type in data:
                #     if type not in labels:
                #         labels.append(type)
                ####
                one_hot_encoding_str = str(one_hot_encoding)[1:-1]
                file.write(f"{n}, {one_hot_encoding_str}\n")
                f.close()
            if n == i:
                break

if __name__ == "__main__":
    json_to_csv()