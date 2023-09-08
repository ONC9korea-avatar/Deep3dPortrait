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
    image_path = args[0]
    image = cv.imread(image_path)

    landmarks = get_face_landmarks(image).flatten()
    landmark_path = image_path.split('.')[0] + '_landmark.txt'

    with open(landmark_path, 'wt') as f:
        f.write(' '.join(map(str, landmarks)))


if __name__ == '__main__':
    main(*sys.argv[1:])