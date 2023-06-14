import numpy as np
import matplotlib.pyplot as plt
import torch

from PIL import Image, ImageDraw

# parse model configuration
def parse_model_config(path):
    file = open(path,"r") # 파일 읽음
    lines = file.read().split("\n") # 한 줄씩 lines에 넣음
    lines = [x for x in lines if x and not x.startswith("#")] # #으로 시작되지 않는 것들만 lines로 받아옴
    lines = [x.rstrip().lstrip() for x in lines] # whitespace 제거

    module_defs = []
    type_name = None
    for line in lines :
        if line.startswith("["):
            type_name = line[1:-1].rstrip()
            if type_name =="net":
                continue
            module_defs.append({})
            module_defs[-1]["type"] = type_name
            if module_defs[-1]["type"] == "convolutional":
                module_defs[-1]["batch_normalize"] = 0
        else :
            if type_name == "net" :
                continue
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    return module_defs

# parse the YOLOv3 configuration
def parse_hyperparm_config(path):
    file = open(path,"r") # 파일 읽음
    lines = file.read().split("\n") # 한 줄씩 lines에 넣음
    lines = [x for x in lines if x and not x.startswith("#")] # #으로 시작되지 않는 것들만 lines로 받아옴
    lines = [x.rstrip().lstrip() for x in lines] # whitespace 제거

    module_defs = [] # 모듈에 대한 definition 정보
    for line in lines :
        if line.startswith("[") : # 괄호로 시작하면 layer을 구분하는 line이고, 괄호로 시작하지 않으면 attribute를 포함하는 line
            type_name = line[1:-1].rstrip() # 라인의 처음부분 다음부터 끝부분 전까지 type_name에 넣음 
            
            if type_name != "net" : # net에 대한 부분은 layer가 아닌 network에 대한 hyperparm을 설정하는 부분이므로 넘어감
                continue
           
            module_defs.append({}) # dictionary 형식으로 추가
            module_defs[-1]["type"] = type_name
            
            if module_defs[-1]["type"] == "convolutional" : # type이 convolutional일 경우에
                module_defs[-1]["batch_normalize"] = 0 # batch_normalize라는 key를 만들어주고 default 값으로 0을 세팅
        
        else : # attribute를 선언
            if type_name != "net" :
                continue
            key, value = line.split("=") # =를 기준으로 key와 value를 나눠줌
            value = value.strip() # value 문자열 및 공백 제거
            module_defs[-1][key.rstrip()] = value.strip() # key 변수의 key를 만들어주고 그 때 value 값으로 세팅
    
    return module_defs


def get_hyperparam(data):
    for d in data :
        if d["type"] =="net" :
            batch = int(d["batch"])
            subdivisions = int(d["subdivisions"])
            momentum = float(d["momentum"])
            decay = float(d["decay"])
            saturation = float(d["saturation"])
            lr = float(d["learning_rate"])
            burn_in = int(d["burn_in"])
            max_batch = int(d["max_batches"])
            lr_policy = d["policy"]
            in_width = int(d["width"])
            in_height = int(d["height"])
            in_channels = int(d["channels"])
            classes = int(d["class"])
            ignore_class = int(d["ignore_cls"])

            return {"batch" : batch,
                    "subdivisions" : subdivisions,
                    "momentum" : momentum,
                    "decay" : decay,
                    "saturation" : saturation,
                    "lr" : lr,
                    "burn_in" : burn_in,
                    "max_batch" : max_batch,
                    "lr_policy" : lr_policy,
                    "in_width" : in_width,
                    "in_height" : in_height,
                    "in_channels" : in_channels,
                    "classes" : classes,
                    "ignore_class" : ignore_class}
        else :
            continue
        
def xywh2xyxy_np(x : np.array): 
    y = np.zeros_like(x)
    y[...,0] = x[...,0] - x[...,2] / 2 # minx
    y[...,1] = x[...,1] - x[...,3] / 2 # miny
    y[...,2] = x[...,0] + x[...,2] / 2 # maxx
    y[...,3] = x[...,1] + x[...,3] / 2 # maxy
    return y

def drawBox(img):
    img = img * 255 # normalize 되어있으므로

    if img.shape[0] == 3:
        img_data = np.array(np.transpose(img, (1,2,0)), dtype=np.uint8)
        img_data = Image.fromarray(img_data)

    # draw = ImageDraw.Draw(img_data)
    plt.imshow(img_data)
    plt.show()

# box_a, box_b IOU
def bbox_iou(box1, box2, xyxy=False, eps = 1e-9):
    box2 = box2.T

    if xyxy:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
        b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
        b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    #intersection
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
    #union
    b1_w, b1_h = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    b2_w, b2_h = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = b1_w * b1_h + b2_w * b2_h - inter + eps

    iou = inter / union

    return iou

def get_lr(optimizer) :
    for param_group in optimizer.param_groups :
        return param_group["lr"]