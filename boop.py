import sys
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

def isolate(img_fn, model_fn, point_x, point_y):
    img = cv2.imread(img_fn)
    input_point = np.array([[point_x, point_y]])
    input_label = np.array([1])
    sam = sam_model_registry["default"](checkpoint=model_fn)
    predictor = SamPredictor(sam)
    predictor.set_image(img)
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
    )

    for i, mask in enumerate(masks):
        mask = (mask*255).astype(np.uint8)

        background = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
        cv2.imwrite("background{}.png".format(i), background)

        # foreground needs transparency
        foreground = cv2.bitwise_and(img, img, mask=mask)
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2BGRA)
        foreground[:,:,3] = mask
        cv2.imwrite("foreground{}.png".format(i), foreground)

def tmp():
    background = cv2.imread("background2.png")
    foreground = cv2.imread("foreground2.png")
    text = cv2.imread("text.png")

    text2 = cv2.cvtColor(text, cv2.COLOR_RGB2GRAY)
    _, text3 = cv2.threshold(text2, 120, 255, cv2.THRESH_BINARY)
    background2 = cv2.bitwise_or(background, background, mask=text3)

    final = cv2.add(background2, foreground)
    cv2.imwrite("final2.png")

    breakpoint()

tmp()
'''
isolate(
    sys.argv[2],
    sys.argv[1],
    sys.argv[3],
    sys.argv[4]
)
'''
