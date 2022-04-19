# import layoutparser as lp
# import cv2

# model = lp.Detectron2LayoutModel(
#             config_path ='lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config', # In model catalog
#             label_map   ={1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"}, # In model`label_map`
#             extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # Optional
#         )


# im = cv2.imread('./data/prima/Images/00000085.tif')

# layout_predicted = model.detect(im)

import src.layoutparser.models.detectron2.layoutmodel as lp
import src.layoutparser.visualization as viz
import cv2

model = lp.Detectron2LayoutModel(
            config_path ='lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config', # In model catalog
            label_map   ={1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"}, # In model`label_map`
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # Optional
        )

im = cv2.imread('./images/00000190.tif')

prediction = model.detect(im)

visualize = viz.draw_box(im, prediction)



print('hello')

