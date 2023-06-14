import os, sys
import torch
import argparse

from torch.utils.data.dataloader import DataLoader
from utils.tools import *
from dataloader.yolodata import *
from dataloader.data_transform import *
from model.yolov3 import *
from train.trainer import *

from tensorboardX import SummaryWriter



def parse_args():
    parser = argparse.ArgumentParser(description="YOLOV3_PYTORCH arguments")
    parser.add_argument("--gpus", type=int, nargs="+", default=[], help="List of GPU device id") # gpu/cpu를 옵션으로 설정
    parser.add_argument("--mode", type=str, default=None, help="mode : train / eval / demo") # mode를 옵션으로 설정
    parser.add_argument("--cfg", type=str, default=None, help="model config path") # configuration 파일 추가
    parser.add_argument("--chekcpoint", type=str, default=None, help="model checkpoint path") # iteration이 멈출 때가 있는데, checkpoint를 통해 학습 중간 중간마다 저장함
    
    if len(sys.argv) == 1 : # argument를 아무 것도 안 줬을 때 예외처리 (python main.py만 입력하면 값이 1이 나옴)
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def collate_fn(batch) :
    batch = [data for data in batch if data is not None]
    # skip invalid data
    if len(batch) == 0 :
        return
    
    imgs, targets, anno_path = list(zip(*batch))
    
    imgs = torch.stack([img for img in imgs]) # 3차원 -> 4차원

    for i, boxes in enumerate(targets) :
        # insert index of batch
        boxes[:,0] = i 
    targets = torch.cat(targets,0)
    
    return imgs, targets, anno_path


def train(cfg_param = None, using_gpus = None): # 모델 학습할 때
    print("train")

    # data loader 6081 images / batch : 4
    my_transform = get_transformations(cfg_param=cfg_param, is_train=True)
    train_data = Yolodata(is_train=True, 
                        transform = my_transform, 
                        cfg_param = cfg_param)
    train_loader = DataLoader(train_data,
                              batch_size = cfg_param["batch"],
                              num_workers = 0, # 데이터 로드 멀티 프로세싱, 데이터 로딩에 사용할 worker 수 지정
                              pin_memory = True, # GPU 메모리에 데이터를 고정할지 여부를 지정하는 것으로, GPU를 사용하는 경우 True로 설정시 데이터가 CPU와 GPU 간에 더 빠르게 복사되어 학습 속도 향상
                              drop_last = True, # 마지막 배치의 크기가 batch_size보다 작을 경우 해당 배치를 무시함
                              shuffle = True,
                              collate_fn = collate_fn)


    model = Darknet53(args.cfg, cfg_param, training = True)
    model.train()
    model.initialize_weights()

    print("GPU : ", torch.cuda.is_available())
        # set device
    if torch.cuda.is_available(): # 하드웨어 환경이 gpu를 사용할 수 있으면
        device = torch.device("cuda:0") # gpu가 1개 있으면 gpu id가 0이 되어서 다음과 같이 설정 (gpu가 2개 있으면 0,1)
    else :  # # 하드웨어 환경이 gpu를 사용할 수 없으면
        device = torch.cuda("cpu")
    model = model.to(device) # 만든 모델을 설정한 device에서 돌아가게 함

    # # load checkpoint
    # # If checkpoint is existed, load the previous checkpoint.
    # if args.checkpoint is not None:
    #     print("load pretrained model ", args.checkpoint)
    #     checkpoint = torch.load(args.checkpoint, map_location = device)
    #     print(checkpoint)


    torch_writer = SummaryWriter("./output")

    trainer = Trainer(model = model, train_loader=train_loader, eval_loader=None, hparam=cfg_param, device = device, torch_writer = torch_writer)
    trainer.run()

def eval(cfg_param = None, using_gpus = None): 
    print("evaluation")

def demo(cfg_param = None, using_gpus = None): 
    print("demo")



if __name__ == "__main__" :
    print("main")
    
    args = parse_args()
    # print("args : ", args)
    # print(args.mode, args.gpus)

    # cfg parser (train, eval, demo에서 공통으로 필요로 함)
    net_data = parse_hyperparm_config(args.cfg)
    cfg_param = get_hyperparam(net_data)
    print(cfg_param)
    
    if args.mode == "train" : # training
        train(cfg_param = cfg_param)
    elif args.mode == "eval" : # evaluation
        eval(cfg_param = cfg_param)
    elif args.mode == "demo" : # demo
        demo(cfg_param = cfg_param)
    else :
        print("unknown mode")
    
    print("finish")