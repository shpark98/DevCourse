from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import sys

def show_img(img_data, text):
    _img_data = img_data * 255
    
    print(_img_data.shape)
    # 4D -> 2D
    _img_data = np.array(_img_data[0,0], dtype=np.uint8)
    
    img_data = Image.fromarray(_img_data)
    draw = ImageDraw.Draw(img_data)
    
    cx, cy = _img_data.shape[0] / 2, _img_data.shape[1] /2
    
    #draw text in img
    if text is not None:
        draw.text((cx,cy), text, fill="red")

    img_data.show()
    img_data.save("res.jpg")
    
    