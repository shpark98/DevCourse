import torch
import os, sys
import numpy as np
import torchvision
from torch.utils.data import Dataset
from PIL import Image


class Yolodata(Dataset):

    # formal path
    file_dir = "" 
    anno_dir = ""
    file_txt = ""

    # train dataset path
    train_dir = "C:\\Users\\wendy\\Downloads\\YOLO\\KITTI\\training"
    train_txt = "train.txt"

    # eval dataset path
    valid_dir = "C:\\Users\\wendy\\Downloads\\YOLO\\KITTI\\eval"
    valid_txt = "eval.txt"
    class_str = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
    num_class = None # 클래스 수
    img_data = [] # image data 들어갈 리스트

    def __init__(self, is_train=True, transform = None, cfg_param=None):
        super(Yolodata, self).__init__()
        self.is_train = is_train
        self.transform = transform
        self.num_class = cfg_param["classes"]

        if self.is_train : # train일 경우
            self.file_dir = self.train_dir + "\\JPEGImages\\"                   
            self.file_txt = self.train_dir + "\\ImageSets\\" + self.train_txt
            self.anno_dir = self.train_dir + "\\Annotations\\"
        else :
            self.file_dir = self.valid_dir + "\\JPEGImages\\"                   
            self.file_txt = self.valid_dir + "\\ImageSets\\" + self.valid_txt
            self.anno_dir = self.valid_dir + "\\Annotations\\"

        img_names = []
        img_data = []
        with open(self.file_txt, 'r', encoding ="UTF-8", errors="ignore") as f: # 읽기 전용으로 file_txt를 읽음 
            img_names = [i.replace("\n","") for i in f.readlines()] # file_txt를 한줄씩 받아오는데 줄바꿈부분을 공백으로 replace 함

        for i in img_names:
            if os.path.exists(self.file_dir + i + ".jpg") : # file_dir + 파일명.jpg가 시스템상에 존재하는 경로인지 확인
                img_data.append(i + ".jpg")
            elif os.path.exists(self.file_dir + i + ".JPG") :
                img_data.append(i + ".JPG")
            elif os.path.exists(self.file_dir + i + ".png") :
                img_data.append(i + ".png")
            elif os.path.exists(self.file_dir + i + ".PNG") :
                img_data.append(i + ".PNG")
        
        self.img_data = img_data
        print("data length : {}".format(len(self.img_data)))

    # get item per one element in one batch
    def __getitem__(self, index) : # init에서 파일 경로에서 실제 이미지를 load하고 anotation 파일을 읽어서 학습 input으로 넣음
        img_path = self.file_dir + self.img_data[index] # 해당 인덱스 정보로 파일명을 가져옴

        with open(img_path, "rb") as f :
            img = np.array(Image.open(img_path).convert("RGB"), dtype = np.uint8) # 이미지를 불러옴
            
            img_origin_h, img_origin_w = img.shape[:2] # image shape : [H,W,C]

        if os.path.isdir(self.anno_dir) : # 시스템에 annotation dir 경로가 실제로 있는지 확인
            txt_name = self.img_data[index]       # annotation file을 불러옴
            for ext in [".png", ".PNG", ".jpg", ".JPG"] :
                txt_name = txt_name.replace(ext, ".txt") # txt_name이 annotation file 포맷으로 변환이 됨
            anno_path = self.anno_dir + txt_name

            if not os.path.exists(anno_path) :
                return 
            
            bbox = [] # 바운딩 박스 [class, center_x, center_y, width, height]
            with open(anno_path, "r") as f :
                for line in f.readlines() : # annotation 파일을 한 줄씩 읽음
                    line = line.replace("\n", "")
                    gt_data = [l for l in line.split(" ")]

                    if len(gt_data) < 5: # 읽어들인 파일이 5개 이하의 값을 읽어들일 경우 예외처리 (비정상적인 데이터)
                        continue
                    cls, cx, cy, w, h = float(gt_data[0]), float(gt_data[1]), float(gt_data[2]), float(gt_data[3]), float(gt_data[4])
                    bbox.append([cls, cx, cy, w, h])
                        
            bbox = np.array(bbox)
            empty_target = False

            # bbox가 잘못 되었을 경우 예외처리
            if bbox.shape[0] == 0:
                empty_target = True
                bbox = np.array([0,0,0,0,0])

            # data augmentation : 데이터가 부족할 때 갖고있는 데이터를 다양한 방법으로 다른 이미지의 타입으로 만들어 학습에 성능을 올림
            if self.transform is not None:
                img, bbox = self.transform((img, bbox))

            if not empty_target:
                batch_idx = torch.zeros(bbox.shape[0])
                target_data = torch.cat((batch_idx.view(-1,1), torch.tensor(bbox)), dim=1)
            else :
                return
            return img, target_data, anno_path
        
        else : # 시스템에 annotation dir 경로가 실제로 없을 경우
            bbox = np.array([[0,0,0,0,0]])
            if self.transform is not None:
                img, _ = self.transform((img,bbox))  # 이미지와 바운딩 박스를 입력으로 받아 처리한 후, 변환된 이미지와 새로운 바운딩 박스를 반환합니다.
            return img, None, None

    def __len__(self) :
        return len(self.img_data)