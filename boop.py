import sys
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import subprocess


def render_text(text_in_fn, text_out_fn, size_y, size_x, font, font_size):
    subprocess.run([
        "convert",
        "-size",
        "{}x{}".format(size_x, size_y),
        "xc:white",
        "-font",
        font,
        "-pointsize",
        str(font_size),
        "-fill",
        "black",
        "-draw",
        "@{}".format(text_in_fn),
        text_out_fn,
    ])
    return cv2.imread(text_out_fn)


def run(model_fn, img_fn, text_fn, point_y, point_x, font, font_size):
    # read img
    img = cv2.imread(img_fn)

    # read text and render it to image
    size_y = img.shape[0]
    size_x = img.shape[1]
    text_png = "text.png"
    text_img = render_text(text_fn, text_png, size_y, size_x, font, font_size)

    # load model
    predictor = SamPredictor(sam_model_registry["default"](checkpoint=model_fn))

    # generate masks
    predictor.set_image(img)
    input_point = np.array([[point_y, point_x]])
    input_label = np.array([1])
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
    )

    for mask_count, mask in enumerate(masks):

        # separate fore/background
        mask = (mask*255).astype(np.uint8)
        background = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
        background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
        # foreground needs transparency
        foreground = cv2.bitwise_and(img, img, mask=mask)
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2BGRA)
        foreground[:,:,3] = mask

        # combine layers
        text2 = cv2.cvtColor(text_img, cv2.COLOR_RGB2GRAY)
        _, text3 = cv2.threshold(text2, 120, 255, cv2.THRESH_BINARY)
        background2 = cv2.bitwise_or(background, background, mask=text3)
        final = cv2.add(background2, foreground)

        # save
        img_fn_last_dot = img_fn.rfind(".")
        output_fn = "{}_final{}.{}".format(img_fn[:img_fn_last_dot], mask_count, img_fn[img_fn_last_dot+1:])
        cv2.imwrite(output_fn, final)

run("../sam_vit_h_4b8939.pth", "cercei1.jpg", "text.txt", 1900, 1600, "Ubuntu", 200)
'''
render_text("text.txt", "text.png", 2448, 3264, "Ubuntu", 200)
'''
