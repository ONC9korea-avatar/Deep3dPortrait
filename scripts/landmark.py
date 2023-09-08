import face_alignment
import cv2 as cv

import numpy as np
import os, sys

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D)

def get_face_landmarks(image):
    h, w, *_ = image.shape
    pred = fa.get_landmarks(image)[0]

    landmarks = np.stack([pred[:,0], h - 1 - pred[:,1]], axis=1)
    return landmarks

def main(*args):
    data_path = args[0]
    image_name = [i for i in os.listdir(data_path)
                  if i.endswith('.png') or i.endswith('.jpg') or i.endswith('.jpeg')][0]
    image = cv.imread(os.path.join(data_path, image_name))

    landmarks = get_face_landmarks(image).flatten()
    landmark_name = image_name.split('.')[0] + '_landmark.txt'

    with open(os.path.join(data_path, landmark_name), 'wt') as f:
        f.write(' '.join(map(str, landmarks)))


if __name__ == '__main__':
    main(*sys.argv[1:])