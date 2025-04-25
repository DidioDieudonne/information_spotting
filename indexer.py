import numpy as np
import os

def build_index(descriptor_folder):
    index = {}
    for file in os.listdir(descriptor_folder):
        if file.endswith('.npy'):
            doc_name = file.replace('.npy', '')
            descriptors = np.load(os.path.join(descriptor_folder, file))
            index[doc_name] = descriptors
    return index