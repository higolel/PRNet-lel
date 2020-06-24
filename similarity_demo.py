import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
import cv2
from utils.cv_plot import plot_kpt

from api import PRN

from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from utils.write import write_obj_with_colors, write_obj_with_texture

def process_image(image_path, isDlib, prn):
    image = imread(image_path)
    [h, w, c] = image.shape
    if c>3:
        image = image[:,:,:3]

        # the core: regress position map
    if isDlib:
        max_size = max(image.shape[0], image.shape[1])
        if max_size> 1000:
            image = rescale(image, 1000./max_size)
            image = (image*255).astype(np.uint8)
        pos = prn.process(image) # use dlib to detect face
    else:
        if image.shape[0] == image.shape[1]:
            image = resize(image, (256,256))
            pos = prn.net_forward(image/255.) # input image has been cropped to 256x256
        else:
            box = np.array([0, image.shape[1]-1, 0, image.shape[0]-1]) # cropped with bounding box
            pos = prn.process(image, box)

    image = image/255.
    return pos


def main(args):
    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU
    prn = PRN(is_dlib = args.isDlib)

    # ------------- load data
    pos1 = process_image(args.inputImage1, args.isDlib, prn)
    pos2 = process_image(args.inputImage2, args.isDlib, prn)

    kpt1 = prn.get_landmarks(pos1)
    kpt2 = prn.get_landmarks(pos2)

    dot = sum([a * b for a, b in zip(kpt1, kpt2)])
    denom = np.linalg.norm(kpt1) * np.linalg.norm(kpt2)
    result = dot / denom
    print(result)
    #dot = np.sum(np.multiply(kpt1, kpt2), axis=1)
    #norm = np.linalg.norm(kpt1, axis=1) * np.linalg.norm(kpt2, axis=1)
    #dist = np.mean(dot / norm)
    #print(dist)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('--inputImage1', default = 'image1', type = str,
                        help = 'path to the input directory, where input images are stored.')
    parser.add_argument('--inputImage2', default = 'image2', type = str,
                        help = 'path to the input directory, where input images are stored.')
    parser.add_argument('--gpu', default = '0', type = str,
                        help = 'set gpu id, -1 for CPU')
    parser.add_argument('--isDlib', default = True, type = ast.literal_eval,
                        help='whether to use dlib for detecting face, default is True, if False, the input image should be cropped in advance')
    parser.add_argument('--is3d', default = True, type=ast.literal_eval,
                        help = 'whether to output 3D face(.obj). default save colors.')
    parser.add_argument('--isMat', default = False, type = ast.literal_eval,
                        help = 'whether to save vertices,color,triangles as mat for matlab showing')
    parser.add_argument('--isKpt', default = True, type = ast.literal_eval,
                        help = 'whether to output key points(.txt)')
    main(parser.parse_args())
