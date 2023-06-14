import numpy as np
import cv2
import torch
from torchvision import transforms as tf

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage # augmentation을 할 때 바운딩 박스의 값들을 조정함

from utils.tools import *

def get_transformations(cfg_param = None, is_train = None):
    if is_train : # 학습할 때만 augmentation을 사용함
        data_transform = tf.Compose([AbsoulteLabels(),
                                     DefaultAug(),
                                     RelativeLabels(),
                                     ResizeImage(new_size=(cfg_param["in_width"], cfg_param["in_height"])),
                                     ToTensor()])
    else :
        data_transform = tf.Compose([AbsoulteLabels(),
                                     ResizeImage(new_size=(cfg_param["in_width"], cfg_param["in_height"])),
                                     RelativeLabels(),
                                     ToTensor()])
    return data_transform

# absolute bbox : normalize된 바운딩박스의 값들을 절대값으로 바꿔줘야 함 -> augmentation을 할 때 절대값을 갖고 있어야 오리지널 input 이미지에 대한 정보를 잃지 않음
class AbsoulteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self,data) : 
        image, label = data
        h, w, _ = image.shape
        label[:,[1,3]] *= w # 1 : cx, 3 : w 
        label[:,[2,4]] *= h # 2 : cy, 4 : h
        return image, label

class RelativeLabels(object):
    def __init__(self,):
        pass

    def __call__(self,data):
        image, label = data
        h, w, _ = image.shape
        label[:,[1,3]] /= w # 1 : cx, 3 : w 
        label[:,[2,4]] /= h # 2 : cy, 4 : h
        return image, label
    

# numpyarray2tensor 
class ToTensor(object):
    def __init__(self,):
        pass
    
    def __call__(self, data):
        image, label = data
        image = torch.tensor(np.transpose(np.array(image, dtype =float) / 255, (2,0,1)), dtype =torch.float32) # HWC -> CHW
        label = torch.FloatTensor(np.array(label)) # 32비트 부동 소수점 텐서로 변환
        
        return image, label
        
class ResizeImage(object):
    def __init__(self, new_size, interpolation=cv2.INTER_LINEAR):
        self.new_size = tuple(new_size)
        self.interpolation = interpolation

    def __call__(self, data):
        image, label = data
        image = cv2.resize(image, self.new_size, interpolation = self.interpolation)
        return image, label

class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self,data):
        # unpack data
        img, labels = data

        # convert xywh to xyxy(minx miny maxx maxy) imgaug에서 이러한 형식으로 사용함
        boxes = np.array(labels)
        boxes[:,1:] = xywh2xyxy_np(boxes[:,1:])

        # convert bbox to imgaug format
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*box[1:], label=box[0]) for box in boxes],  # 1~4 xyxy 값과 클래스 정보를 넣어줌
            shape = img.shape)

        # apply augmentations
        img, bounding_boxes = self.augmentations(image=img, 
                                                 bounding_boxes = bounding_boxes)
        
        bounding_boxes = bounding_boxes.clip_out_of_image() # 이미지 밖으로 나가는 bounding box가 있을 때 clip 예외처리

        # convert bounding_boxes to np.array()
        boxes = np.zeros((len(bounding_boxes), 5))
        for box_idx, box in enumerate(bounding_boxes):
            x1, y1, x2, y2= box.x1, box.y1, box.x2, box.y2

            # return xyxy2xywh [x,y,w,h]
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = (x1 + x2) / 2
            boxes[box_idx, 2] = (y1 + y2) / 2
            boxes[box_idx, 3] = x2 - x1 
            boxes[box_idx, 4] = y2 - y1

        return img, boxes
    
class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Sharpen(0.0, 0.1),
            iaa.Affine(rotate=(-0,0), translate_percent=(-0.1,0.1), scale=(0.8,1.5))])
        