# coding=utf-8
import os
import sys
import random
import tensorflow as tf
import json
from PIL import Image
import csv

DIRECTORY_IMAGES_1 = 'train'
DIRECTORY_IMAGES_2 = 'test'
DIRECTORY_IMAGES_3 = 'other'
RANDOM_SEED = 4242


def _process_image(directory, name, writer, directory_images):
    filename = os.path.join(directory, directory_images, name + '.jpg')
    filedir = directory + "/annotations.json"
    annos = json.loads(open(filedir).read())
    red_round_labels = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15',
                        'p16', 'p17', 'p18',
                        'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'pa10', 'pb', 'pc',
                        'pd',
                        'pe', 'ph3.5', 'pl40', 'pm10', 'pn', 'pne', 'pnl', 'pw3']
    with Image.open(filename) as img:
        shape = [img.height,img.width,3]
    if annos['imgs'][name]['objects']:
        for obj in annos['imgs'][name]['objects']:
            label = obj['category']
            if label in red_round_labels:
                label = 1
            else:
                label= 0
            bbox = obj['bbox']
            ymin = float(bbox['ymin'])
            xmin = float(bbox['xmin'])
            ymax = float(bbox['ymax'])
            xmax = float(bbox['xmax'])
            line=filename+' '+str(xmin)+' '+str(ymin)+' '+str(xmax)+' '+str(ymax)+' ' + str(label)+'\n'
            writer.write(line)
    return


def run(tt100k_root, split, writer, directory_images):
    split_file_path = os.path.join(tt100k_root, split, 'ids.txt')
    print('>> ', split_file_path)
    i = 0
    with open(split_file_path) as f:
        filenames = f.readlines()
    while i < len(filenames):
        sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(filenames)))
        sys.stdout.flush()
        filename = filenames[i].strip()
        _process_image(tt100k_root, filename, writer, directory_images)
        i += 1
    print('\n>> Finished converting the TT100K %s dataset!' % split)


if __name__ == '__main__':
    writer = open('test.txt','a')
    #run('../data', 'train', writer, DIRECTORY_IMAGES_1)
    #run('../data', 'test', writer, DIRECTORY_IMAGES_2)
    run('../data', 'other', writer, DIRECTORY_IMAGES_3)
    writer.close()
