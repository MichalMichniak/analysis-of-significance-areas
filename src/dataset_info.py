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

DICT_LABELS_ENCODING = {i:n for n,i in enumerate(LABELS_DS)}
DICT_LABELS_DECODING = {n:i for n,i in enumerate(LABELS_DS)}