import os
import cv2
import numpy as np
from matplotlib import image
from PIL        import ImageDraw
import matplotlib.pyplot as plt


import src.ocr.tesseract_agent as tes
from src.elements import Rectangle, TextBlock, Layout


#results is 2d prediction, classes is label map
def bbox_yield(result, label_map):
    layout = Layout()

    for i in range(1,len(list(label_map))):
        arr = result != i

        plt.imsave('./filename.png', arr, cmap='binary')
        img = cv2.imread('./filename.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        
        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)
            layout.append(TextBlock(Rectangle(x,y,x+w,y+h), type=label_map[i]))
        
        os.remove('./filename.png')
    
    return layout

def class_of_interest(layout, roi):
    text_blocks  = Layout([b for b in layout if b.type==roi])
    other_blocks = Layout([b for b in layout if b.type!=roi])
    return Layout([b for b in text_blocks if not any(b.is_in(b_fig) for b_fig in other_blocks)]), other_blocks

def remove_empty_rectangles(image, layout, threshold=50):
    ocr_agent = tes.TesseractAgent(languages='eng') 

    arr_im = np.asarray(image)
    new_layout = Layout()

    for block in layout:
        segment_image = (block.crop_image(arr_im))

        text = ocr_agent.detect(segment_image)
        if len(text) >= threshold: 
            new_layout.append(block)
    
    return new_layout

def mask_other_regions(image, blocks): #not used in the solution
    draw = ImageDraw.Draw(image)
    for i in blocks:
        draw.rectangle([i.block.x_1, i.block.y_1, i.block.x_2, i.block.y_2], fill=(255,255,255))
    return image

def ocr(image, layout):
    ocr_agent = tes.TesseractAgent(languages='eng')

    arr_im = np.asarray(image)
    for block in layout:
        segment_image = (block.crop_image(arr_im))

        text = ocr_agent.detect(segment_image)
        block.set(text=text, inplace=True)

    for txt in layout.get_texts():
        print(txt, end='\n---\n')

    return layout
