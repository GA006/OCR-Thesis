import os, re, json
import imagesize
from glob import glob
from bs4 import BeautifulSoup
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import sys
sys.path.append('..')
import cocosplit

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)



def cvt_coords_to_array(obj):
    
    return np.array(
            [(float(pt['x']), float(pt['y']))
                 for pt in obj.find_all("Point")]
        )

def cal_ployarea(points):
    x = points[:,0]
    y = points[:,1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def _create_category(schema=0):

    if schema==0:
        
        categories = \
            [{"supercategory": "layout", "id": 0, "name": "Background"},
             {"supercategory": "layout", "id": 1, "name": "Paragraph"},
             {"supercategory": "layout", "id": 2, "name": "Other"},]
        
        find_categories = lambda name: \
            [val["id"] for val in categories if val['name'] == name][0]
        
        conversion = \
            {
                'paragraph':        find_categories("Paragraph"),
                'heading':          find_categories("Other"),
                'header':           find_categories("Other"),
                'footer':           find_categories("Other"),
                'page-number':      find_categories("Other"),
                'floating':         find_categories("Other"),
                'credit':           find_categories("Other"),
                'caption':          find_categories("Other"),
                'drop-capital':     find_categories("Other"),
                'TableRegion':      find_categories("Other"),
                'MathsRegion':      find_categories("Other"),
                'ChartRegion':      find_categories("Other"),
                'GraphicRegion':    find_categories("Other"),
                'ImageRegion':      find_categories("Other"),
                'LineDrawingRegion':find_categories("Other"),
                'SeparatorRegion':  find_categories("Other"),
                'NoiseRegion':      find_categories("Other"),
                'FrameRegion':      find_categories("Other"),
            }
        
        return categories, conversion

_categories, _categories_conversion = _create_category(schema=0)

_info = {
    "description": "PRIMA Layout Analysis Dataset",
    "url": "https://www.primaresearch.org/datasets/Layout_Analysis",
    "version": "1.0",
    "year": 2010,
    "contributor": "PRIMA Research",
    "date_created": "2020/09/01",
}

def _load_soup(filename):
    with open(filename, "r") as fp:
        soup = BeautifulSoup(fp.read(),'xml')
    
    return soup

def _image_template(image_id, image_path):
    
    width, height = imagesize.get(image_path)
    
    return {
        "file_name": os.path.basename(image_path),
        "height": height,
        "width": width,
        "id": int(image_id)
    }
    
def _anno_template(anno_id, image_id, pts, obj_tag):

    x_1, x_2 = pts[:,0].min(), pts[:,0].max()
    y_1, y_2 = pts[:,1].min(), pts[:,1].max()
    height = y_2 - y_1
    width  = x_2 - x_1
    
    return {
        "segmentation": [pts.flatten().tolist()],
        "area": cal_ployarea(pts),
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": [x_1, y_1, width, height],
        "category_id": _categories_conversion[obj_tag],
        "id": anno_id
    }

class PRIMADataset():
    
    def __init__(self, base_path, anno_path='XML',
                                  image_path='Images'):
        
        self.base_path = base_path
        self.anno_path = os.path.join(base_path, anno_path)
        self.image_path = os.path.join(base_path, image_path)
        
        self._ids = self.find_all_image_ids()
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        return self.load_image_and_annotaiton(idx)
    
    def find_all_annotation_files(self):
        return glob(os.path.join(self.anno_path, '*.xml'))
    
    def find_all_image_ids(self):
        replacer = lambda s: os.path.basename(s).replace('pc-', '').replace('.xml', '')
        return [replacer(s) for s in self.find_all_annotation_files()]
    
    def load_image_and_annotaiton(self, idx):
        
        image_id = self._ids[idx]
        
        image_path = os.path.join(self.image_path, f'{image_id}.tif')
        image = Image.open(image_path)
        
        anno = self.load_annotation(idx)
        
        return image, anno

    def load_annotation(self, idx):
        image_id = self._ids[idx]

        anno_path  = os.path.join(self.anno_path,  f'pc-{image_id}.xml')
        # A dirtly hack to load the files w/wo pc- simualtaneously
        if not os.path.exists(anno_path):
            anno_path = os.path.join(self.anno_path,  f'{image_id}.xml')
            assert os.path.exists(anno_path), "Invalid path"
        anno = _load_soup(anno_path)

        return anno



    def convert_to_COCO(self, save_path):
        
        all_image_infos = []
        all_anno_infos  = []
        anno_id = 0
        
        for idx, image_id in enumerate(tqdm(self._ids)):
            
            # We use the idx as the image id
            
            image_path = os.path.join(self.image_path, f'{image_id}.tif')
            image_info = _image_template(idx, image_path)
            all_image_infos.append(image_info)
            
            anno = self.load_annotation(idx)

            for item in anno.find_all(re.compile(".*Region")):

                pts = cvt_coords_to_array(item.Coords)
                if 0 not in pts.shape:
                    # Sometimes there will be polygons with less
                    # than 4 edges, and they could not be appropriately
                    # handled by the COCO format. So we just drop them.
                    if pts.shape[0] >= 4:
                        # NOTE: Fork-custom
                        if item.name == 'TextRegion':
                            if not item.attrs.get('type'):
                                item_tag = item.name
                            else:
                                item_tag = item.attrs['type']
                        else:
                            item_tag = item.name

                        try:
                            anno_info = _anno_template(anno_id, idx, pts, item_tag)
                        except KeyError:
                            continue

                        all_anno_infos.append(anno_info)
                        anno_id += 1

        
            
        final_annotation = {
            "info": _info,
            "licenses": [],
            "images": all_image_infos,
            "annotations": all_anno_infos,
            "categories": _categories} 
        
        with open(save_path, 'w') as fp:
            json.dump(final_annotation, fp, cls=NpEncoder)
        
        return final_annotation


parser = argparse.ArgumentParser()

parser.add_argument('--prima_datapath', type=str, default='../data/prima', help='the path to the prima data folders')
parser.add_argument('--anno_savepath',  type=str, default='../data/prima/annotations.json', help='the path to save the new annotations')


if __name__ == "__main__":
    args = parser.parse_args()

    print("Start running the conversion script")
    
    print(f"Loading the information from the path {args.prima_datapath}")
    dataset = PRIMADataset(args.prima_datapath)
    
    print(f"Saving the annotation to {args.anno_savepath}")
    res = dataset.convert_to_COCO(args.anno_savepath)

    cocosplit.main(
        args.anno_savepath,
        split_ratio=0.8,
        having_annotations=True, 
        train_save_path=args.anno_savepath.replace('.json', '-train.json'),
        test_save_path=args.anno_savepath.replace('.json', '-val.json'),
        random_state=24)