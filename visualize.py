#!/usr/bin/env python3
"""
Script to visualize a trained Donkeycar neural network model

Usage:
    visualize.py (--tub=<tub_path) (--model=<model>) [--output=<output>] [--headless]

Options:
    -h --help          Show this screen.
    --tub TUBPATHS     Record tub path
    --model MODELPATH  Trained NN model
    --output VIDEOPATH Video output file path
    --headless         Do not show image window while running
"""
from __future__ import print_function

import numpy as np
import keras
import glob
import re
import json
import traceback

from docopt import docopt

from PIL import Image

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations

import cv2

from pathlib import Path

# input image dimensions
print("TODO: configuration file for image dimensions")
img_rows, img_cols = 240, 100

def get_record_image(base_path, record_path):
    with open(record_path, 'r') as record:
        data = json.load(record)
        img_path = data['cam/image_array']
    img = Image.open('%s/%s' % (base_path, img_path))
    img = np.array(img)
    return img

def visualize(model_path, tub_path, headless=False, output='output.avi'):
    my_file = Path(model_path)
    if my_file.is_file():
        model = keras.models.load_model(model_path)
        print("Saved model found")
    else:
        print("No model found")
        return

    print("TODO: configurable layer name")
    layer_idx = utils.find_layer_idx(model, 'angle_out')

    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)


    records = glob.glob('%s/record*.json' % tub_path)
    count = len(records)

    records = ((int(re.search('.+_(\d+).json', path).group(1)), path) for path in records)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output, fourcc, 20.0, (img_rows, img_cols))

    blend = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX

    i = 0
    start = 0
    try:
        for _, record in sorted(records):
            i = i + 1
            if (i > start):
                image = get_record_image(tub_path, record)
                
                # TODO: configurable backprop_modifier
                grads = visualize_saliency(model, layer_idx, backprop_modifier='guided', filter_indices=None, seed_input=image)
                grads_rgb = grads * 255
                grads_rgb = grads_rgb.astype(np.uint8)
                grads_rgb = cv2.applyColorMap(grads_rgb, cv2.COLORMAP_JET)
                
                grads_float = grads_rgb.astype(np.float)
                blended_image = (blend*image + (1-blend)*grads_float).astype(np.uint8) 
                out.write(blended_image)
                
                # TODO: visualize predictions with a parameter
                # image_arr = image.reshape((1,) + image.shape)
                # angle, throttle = model.predict([image_arr])
                # print(angle[0][0], throttle[0][0])
                # print("Predict ready")
                # cv2.putText(blended_image, str(round(angle[0][0], 2)), (20, 40), font, 0.8, (255, 0, 0), 2)
                if (not headless):
                    cv2.imshow('Saliency', blended_image)
                    cv2.waitKey(10)
                if (i % 10 == 0):
                    print(i, '/', count)
    except KeyboardInterrupt:
        print('Saved', output)
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    args = docopt(__doc__)
    tub_path = args['--tub']
    model_path = args['--model']
    output_path = args['--output']
    headless = args['--headless']
    visualize(model_path, tub_path, output=output_path, headless=headless)

