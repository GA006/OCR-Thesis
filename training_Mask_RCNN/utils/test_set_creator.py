
'''
Takes json, path to images and destination.
Outputs the images which don't have any annotations in the json
file, in the destination folder. In our experiments, we will consider 
the images without annotations as test images.
'''


from importlib.resources import path
import json
import os
import shutil
import argparse

def test_set_creator(path_to_json, data_path, destinaton):
    f = open(path_to_json)
    data = json.load(f)

    test_set = []
    for i in data['images']:
        annot = [j for j in data['annotations'] if i['id'] == j['image_id']]
        if annot == []:
            test_set.append(i['file_name'])
    
    try:
        os.mkdir(destinaton)
    except OSError as error:
        print(error)  

    for i in test_set:
        shutil.copyfile(os.path.join(data_path, i), os.path.join(destinaton, i))


parser = argparse.ArgumentParser()

parser.add_argument('--path_to_json', type=str, default='../data/prima/anno.json')
parser.add_argument('--data_path', type=str, default='../data/prima/Images')
parser.add_argument('--destination', type=str, default='../test_set')

if __name__ == "__main__":
    args = parser.parse_args()
    print('Creating Test Set')
    test_set_creator(args.path_to_json, args.data_path, args.destination)
    print('DONE')