import os
import json
import numpy as np
import shutil
from PIL import Image, ImageDraw

def annotated_images_check(path_to_annotations):
    f    = open(path_to_annotations)
    data = json.load(f)

    ids = [j['image_id'] for j in data['annotations']]
    names = []
    for i in data['images']:
        if i['id'] in ids:
            names.append(i['file_name'])

    return names


def train_set_builder(path_to_images, path_to_save, path_to_annotations):
    files = annotated_images_check(path_to_annotations)
    for filename in files:
        filename_new = filename.split('.')[0] + '.png'
        image = Image.open(os.path.join(path_to_images, filename))
        image.save(os.path.join(path_to_save, filename_new))


def annotation_set_builder(path_to_images, path_to_annotations, path_to_save, palette):
    palette = np.array(palette, dtype=np.uint8)

    f    = open(path_to_annotations)
    data = json.load(f)

    files = annotated_images_check(path_to_annotations)

    for filename in files:
        image_id = [i['id'] for i in data['images'] if filename == i['file_name']][0]
        filename_new = filename.split('.')[0] + '.png'
        image = Image.open(os.path.join(path_to_images, filename_new))
        new = Image.new('P', image.size, 4)
        draw = ImageDraw.Draw(new, 'P')
        draw.rectangle([(0,0), (image.size[0]-1, image.size[1]-1)], outline=0, fill=0)
        for i in data['annotations']:
            if i['image_id'] == image_id:
                if len(i['segmentation']) == 0:
                    continue
                coords = [(i['segmentation'][0][j],i['segmentation'][0][j+1]) for j in range(0,len(i['segmentation'][0]),2)]
                draw.polygon(coords, fill=i['category_id'])
        new.putpalette(palette)
        new.save(os.path.join(path_to_save, filename_new))


def split_set(path_to_tr_ann, path_to_val_ann, path_to_tr_split, path_to_val_split):
    f = open(path_to_tr_ann)
    data_train = json.load(f)

    g = open(path_to_val_ann)
    data_val = json.load(g)
    
    train = [i['file_name'].split('.')[0] for i in data_train['images']]
    val   = [i['file_name'].split('.')[0] for i in data_val['images']]

    with open(path_to_tr_split, 'w') as f:
        f.writelines(line + '\n' for line in train)

    with open(path_to_val_split, 'w') as f:
        f.writelines(line + '\n' for line in val)

