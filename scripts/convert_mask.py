from scipy.io import loadmat, savemat
import cv2 as cv

import numpy as np
import os, sys

def convert_mask_label(face_parsing_mask):
    mask_converted = np.zeros_like(face_parsing_mask, dtype=np.uint8)

    index_1 = ((face_parsing_mask >= 1) & (face_parsing_mask <= 6)) | (face_parsing_mask == 10)
    mask_converted[index_1] = 1

    mask_converted[face_parsing_mask == 17] = 2
    mask_converted[face_parsing_mask == 7] = 3
    mask_converted[(face_parsing_mask == 8) | (face_parsing_mask == 9)] = 4
    
    index_5 = (face_parsing_mask >= 11) & (face_parsing_mask <= 13)
    mask_converted[index_5] = 5

    return mask_converted

    

def main(*args):
    data_path = args[0]
    mask_name = [i for i in os.listdir(data_path)
                  if i.endswith('.png')][0]
    mask = cv.imread(os.path.join(data_path, mask_name), cv.IMREAD_GRAYSCALE)
    # print(mask.shape, type(mask[0][0]))
    # print(np.unique(mask))

    mask_converted = convert_mask_label(mask)
    mat_name = mask_name.split('.')[0] + '.mat'
    savemat(os.path.join(data_path, '..', mat_name), {'mask':mask_converted})


if __name__ == '__main__':
    main(*sys.argv[1:])