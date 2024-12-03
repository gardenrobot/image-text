import sys
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

img = cv2.imread(sys.argv[2])
input_point = np.array([[sys.argv[3], sys.argv[4]]])
input_label = np.array([1])
breakpoint()
sam = sam_model_registry["default"](checkpoint=sys.argv[1])
predictor = SamPredictor(sam)
predictor.set_image(img)
masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
)

for i, mask in enumerate(masks):
    mask = (mask*255).astype(np.uint8)
    foreground = cv2.bitwise_and(img, img, mask=mask)
    background = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    cv2.imwrite("foreground{}.jpg".format(i), foreground)
    cv2.imwrite("background{}.jpg".format(i), background)

