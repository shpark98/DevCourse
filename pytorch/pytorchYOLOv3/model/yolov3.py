import os, sys
import numpy as np
import torch
import torch.nn as nn

from utils.tools import *

def make_conv_layer(layer_idx : int, modules : nn.Module, layer_info : dict, in_channel : int) :
    filters = int(layer_info["filters"]) # output channel size
    size = int(layer_info["size"]) # kernel size
    stride = int(layer_info["stride"]) # stride
    pad = (size - 1) // 2
    modules.add_module("layer_"+str(layer_idx)+"_conv", 
                       nn.Conv2d(in_channel, filters, size, stride, pad))
    
    if layer_info["batch_normalize"] == "1" :
        modules.add_module("layer_"+str(layer_idx)+"_bn", 
                           nn.BatchNorm2d(filters))
    if layer_info["activation"] == "leaky" :
        modules.add_module("layer_"+str(layer_idx)+"_act", 
                            nn.LeakyReLU())
    elif layer_info["activation"] == "relu" :
        modules.add_module("layer_"+str(layer_idx)+"_act", 
                            nn.ReLU())
        
def make_shortcut_layer(layer_idx : int, modules : nn.Module) :
    modules.add_module("layer_"+str(layer_idx)+"_shortcut", nn.Identity())

def make_route_layer(layer_idx : int, modules : nn.Module) :
    modules.add_module("layer_"+str(layer_idx)+"_route", nn.Identity())

def make_upsample_layer(layer_idx : int, modules : nn.Module, layer_info : dict) :
    stride = int(layer_info["stride"])
    modules.add_module("layer_"+str(layer_idx)+"_upsample", nn.Upsample(scale_factor = stride, mode = "nearest"))


class Yololayer(nn.Module) :
    def __init__(self, layer_info : dict, in_width : int, in_height : int, is_train : bool) :
        super(Yololayer, self).__init__()
        self.n_classes = int(layer_info["classes"])
        self.ignore_thresh = float(layer_info["ignore_thresh"])
        self.box_attr = self.n_classes + 5 # bbox[4] + objectness[1] + class_prob[n_classes]
        mask_idxes = [int(x) for x in layer_info["mask"].split(",")]
        anchor_all = [int(x) for x in layer_info["anchors"].split(",")]
        anchor_all = [(anchor_all[i], anchor_all[i+1]) for i in range(0, len(anchor_all), 2)]
        self.anchor = torch.tensor([anchor_all[x] for x in mask_idxes])
        self.in_width = in_width
        self.in_height = in_height
        self.stride = None
        self.lw = None
        self.lh = None
        self.is_train = is_train

    def forward(self, x):
        # x : input, [N C H W]
        self.lw, self.lh = x.shape[3], x.shape[2]
        self.anchor = self.anchor.to(x.device)
        self.stride = torch.tensor([torch.div(self.in_width, self.lw, rounding_mode = "floor"),
                                   torch.div(self.in_height, self.lh, rounding_mode = "floor")]).to(x.device)
        
        # if kitti data n_classes is 8. C = (8 + 5) * 3 = 39
        # [batch, box_attrib * anchor, lw, lh] ex) [1, 39, 19, 19]

        # 4 dim [batch, box_attrib * anchor, lh, lw] -> 5 dim [ batch, anchor, box_attrib, lh, lw] -> [batch, anchor, lh, lw, box_attrib] 
        x = x.view(-1, self.anchor.shape[0], self.box_attr, self.lh, self.lw).permute(0,1,3,4,2).contiguous()

        return x

class Darknet53(nn.Module):
    def __init__(self, cfg, param, training):
        super().__init__()
        self.batch = int(param['batch'])
        self.in_channels = int(param["in_channels"])
        self.in_width = int(param["in_width"])
        self.in_height = int(param["in_height"])
        self.n_classes = int(param["classes"])
        self.module_cfg = parse_model_config(cfg)
        self.module_list = self.set_layer(self.module_cfg)
        self.yolo_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], Yololayer)]
        self.training = training
        
    def set_layer(self, layer_info): # yolov3 모델을 config 파일을 통해 읽어 들임
        module_list = nn.ModuleList() # 세팅할때는 ModuleList()를 사용 -> 각각의 Module들을 Store해줌
        in_channels = [self.in_channels] # First Channels of input
        for layer_idx, info in enumerate(layer_info) :
            modules = nn.Sequential() # Sequential을 사용하여 여러개를 묶어주는 방법으로 사용

            if info["type"] == "convolutional": 
                make_conv_layer(layer_idx, modules, info, in_channels[-1])
                in_channels.append(int(info["filters"]))

            elif info["type"] == "shortcut" :
                make_shortcut_layer(layer_idx, modules)
                in_channels.append(in_channels[-1]) # 이전 채널과 동일

            elif info["type"] == "route" : 
                make_route_layer(layer_idx, modules)
                layers = [int(y) for y in info["layers"].split(",")]
                if len(layers) == 1 :
                    in_channels.append(in_channels[layers[0]])
                elif len(layers) == 2 :
                    in_channels.append(in_channels[layers[0]] + in_channels[layers[1]])

            elif info["type"] == "upsample" : # 
                make_upsample_layer(layer_idx, modules, info)
                in_channels.append(in_channels[-1])

            elif info["type"] == "yolo" :
                yololayer = Yololayer(info, self.in_width, self.in_height, self.training)
                modules.add_module("layer_"+str(layer_idx) + "_yolo", yololayer)
                in_channels.append(in_channels[-1])

            module_list.append(modules)
        return module_list
    
    def initialize_weights(self): # 처음에 학습할 때 weight 값을 잘 주면 실제로 학습이 더 빠르게 수렴될 수 있고 좋은 결과를 얻을 수 있음. 
    # 랜덤 값을 넣어주게 되면 실제로 수렴되는게 어려워지는 상황이 생길 수 있음 따라서 weight 초기화를 진행
        # track all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d) :
                nn.init.kaiming_uniform_(m.weight) # weight 초기화 

                if m.bias is not None :
                    nn.init.constant_(m.bias, 0) # bias는 0으로 만듬
            elif isinstance(m, nn.BatchNorm2d) :
                nn.init.constant_(m.weight, 1) # scale
                nn.init.constant_(m.bias, 0) # shift
            elif isinstance(m, nn.Linear) :
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)



    def forward(self,x) : # type 이름에 따라 각각의 상황에 맞게 만듬
        yolo_result = []
        layer_result = []
        for idx, (name, layer) in enumerate(zip(self.module_cfg, self.module_list)) :
            if name["type"] == "convolutional" :
                x = layer(x)
                layer_result.append(x)
            elif name["type"] == "shortcut" :
                x = x + layer_result[int(name["from"])]
                layer_result.append(x)
            elif name["type"] == "yolo" :
                yolo_x = layer(x)
                layer_result.append(yolo_x)
                yolo_result.append(yolo_x)
            elif name["type"] == "upsample" :
                x = layer(x)
                layer_result.append(x)
            elif name["type"] == "route" : # concat
                layers = [int(y) for y in name["layers"].split(",")]
                x = torch.cat([layer_result[l] for l in layers], dim=1)
                layer_result.append(x)
        return yolo_result