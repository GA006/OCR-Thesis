import src.layoutparser.elements.layout as lay
import numpy as np
import src.layoutparser.elements.layout_elements as le
from PIL import ImageDraw



class PostProcessingSegmentation():

    def __init__(self, blocks, image, masks=None, label_map={1:"Paragraph", 2:"OtherText", 3:"Visual"}):
        self.blocks    = blocks
        self.image     = image
        self.masks     = masks
        self.label_map = label_map
        

    def intersection_over_union(self, a, b):
        x1 = max(a.block.x_1, b.block.x_1)
        y1 = max(a.block.y_1, b.block.y_1)
        x2 = min(a.block.x_2, b.block.x_2)
        y2 = min(a.block.y_2, b.block.y_2)
        if (y2-y1) <= 0 or (x2-x1) <= 0:
            return 0

        intersection = (x2-x1)*(y2-y1)

        areaA = (a.block.x_2 - a.block.x_1) * (a.block.y_2 - a.block.y_1)
        areaB = (b.block.x_2 - b.block.x_1) * (b.block.y_2 - b.block.y_1)

        iou = intersection / (areaA + areaB - intersection)

        return iou

    def unions(self, threshold):
        unions = []
        disappear = []
        count = 1
        bl = self.blocks
        while count != 0:
            count = 0
            for i in range(len(bl)):
                if i in disappear:
                    continue
                not_intersect = True
                for j in range(i+1, len(bl)):
                    if self.intersection_over_union(bl[i], bl[j]) > 0 and abs(bl[i].block.x_1-bl[j].block.x_1) < threshold:
                        unions.append(bl[i].union(bl[j]))
                        disappear.append(j)
                        not_intersect = False
                        count += 1
                if not_intersect:
                    unions.append(bl[i])
            bl = unions
            unions = []
            disappear = []

        return lay.Layout(bl)


    def sorting(self, threshold):
        bl = self.blocks
        arr_im = np.asarray(self.image)

        height, width = arr_im.shape[:2]

        bl.sort(key = lambda b:b.coordinates[0], inplace=True)

        intervals = []

        for i in range(len(bl)):
            x_1 = bl[i].block.x_1
            if i == 0:
                intervals.append(x_1)
                curr_x = x_1
            else:
                if (x_1 - curr_x) >= threshold:
                    intervals.append(x_1)
                    curr_x = x_1  

        blocks_arr = []

        for i in range(len(intervals)):
            if i == len(intervals) - 1:
                curr_interval = le.Interval(intervals[i],width, axis='x').put_on_canvas(self.image)
            else:
                curr_interval = le.Interval(intervals[i], intervals[i+1], axis='x').put_on_canvas(self.image)
            block = bl.filter_by(curr_interval, center=True)
            block.sort(key = lambda b:b.coordinates[1], inplace=True)
            blocks_arr.append(block)

        bl = blocks_arr[0]
        for i in range(1,len(blocks_arr)):
            bl += blocks_arr[i]

        return lay.Layout([b.set(id = idx) for idx, b in enumerate(bl)])


    def left_to_right_overlap(self):
        bl = self.blocks
        for i in range(len(bl)):
            for j in range(i+1, len(bl)):
                if self.intersection_over_union(bl[i], bl[j]) > 0:
                    right_rec_percentage = (bl[i].block.x_2-bl[j].block.x_1) / (bl[i].block.x_2-bl[i].block.x_1)
                    left_rec_percentage  = (bl[i].block.x_2-bl[j].block.x_1) / (bl[j].block.x_2-bl[j].block.x_1)
                    if right_rec_percentage >= left_rec_percentage:
                        bl[j].block.x_1 = bl[i].block.x_2
                    else:
                        bl[i].block.x_2 = bl[j].block.x_1
        
        return lay.Layout(bl)

    def mask_other_regions(self, roi):
        draw = ImageDraw.Draw(self.image)

        for i in self.masks:
            if i[0] != roi:
                coords = np.nonzero(i[1])
                if len(coords[0]) == 0:
                    continue
                coords = [(coords[1][k],coords[0][k]) for k in range(len(coords[0]))]
                draw.polygon(coords)

    def pipeline(self, roi=1, threshold_percent=0.2):

        text_blocks = lay.Layout([b for b in self.blocks if b.type==self.label_map[roi]])
        figure_blocks = lay.Layout([b for b in self.blocks if b.type!=self.label_map[roi]])
        self.blocks = lay.Layout([b for b in text_blocks if not any(b.is_in(b_fig) for b_fig in figure_blocks)])

        if len(self.blocks) == 0:
            raise Exception("NO TEXT BLOCKS")
        self.blocks = lay.Layout([i.pad(left=15, right=15, top=50, bottom=50) for i in text_blocks])

        x_size = np.asarray(self.image).shape[1]

        self.blocks = self.unions(x_size * threshold_percent)

        self.blocks = self.sorting(x_size * threshold_percent)

        self.blocks = self.left_to_right_overlap()

        self.mask_other_regions(roi)

        return self.image, self.blocks






