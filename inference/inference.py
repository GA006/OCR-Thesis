from PIL import Image
import numpy as np
import layoutparser as lp

def inference(model, image_path):
    im = Image.open(image_path)
    layout = model.detect(im)
    lp.draw_box(im, layout, box_width=3, show_element_type=True)
    im.show()

    text_blocks = lp.Layout([b for b in layout if b.type=='TextRegion'])
    figure_blocks = lp.Layout([b for b in layout if b.type=='ImageRegion'])
    text_blocks = lp.Layout([b for b in text_blocks if not any(b.is_in(b_fig) for b_fig in figure_blocks)])

    arr_im = np.asarray(im)

    h, w = arr_im.shape[:2]

    left_interval = lp.Interval(0, w/3, axis='x').put_on_canvas(im)
    left_blocks = text_blocks.filter_by(left_interval, center=True)
    left_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)

    central_interval = lp.Interval(w/3, w*2/3, axis='x').put_on_canvas(im)
    all_central_block = lp.Layout([b for b in text_blocks if b not in left_blocks])
    central_blocks = all_central_block.filter_by(central_interval, center=True)
    central_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)

    right_blocks = lp.Layout([b for b in all_central_block if b not in central_blocks])
    right_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)
    text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + central_blocks + right_blocks)])

    lp.draw_box(im, text_blocks, box_width=3, show_element_id=True)

    ocr_agent = lp.TesseractAgent(languages='eng') 

    for block in text_blocks:
        segment_image = (block.pad(left=5, right=5, top=5, bottom=5).crop_image(arr_im))

        text = ocr_agent.detect(segment_image)
        block.set(text=text, inplace=True)
    
    for txt in text_blocks.get_texts():
        print(txt, end='\n---\n')