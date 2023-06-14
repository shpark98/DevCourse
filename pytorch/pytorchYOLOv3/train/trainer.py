import os, sys
import torch
import torch.optim as optim

from utils.tools import *
from train.loss import *

class Trainer :
    def __init__(self, model, train_loader, eval_loader, hparam, device, torch_writer) : # eval_loader는 학습하는 것이 아닌 평가하는 경우에 사용
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_batch = hparam["max_batch"]
        self.device = device
        self.epoch = 0
        self.iter = 0
        self.yololoss = Yololoss(self.device, self.model.n_classes)
        self.optimizer = optim.SGD(model.parameters(), lr = hparam["lr"], momentum =hparam["momentum"])

        # 학습을 진행할수록 learning rate가 떨어져야 더 정교하게 weight를 업데이트할 수 있음
        # multistepLR을 사용하면 떨어지는 빈도를 multistep 기준으로 만들어줌
        self.schedular_multistep = optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                             milestones = [20, 40, 60], # milestone : 몇 개의 iteration, epoch 에서 learning rate를 감소시킬지
                                                             gamma = 0.5) # gamma : learning rate를 감소 시킬때 얼마의 비율로 감소시킬지
        
        self.torch_writer = torch_writer
        
    def run_iter(self) : # 한 번 epoch 안의 iter 단위의 for loop
        for i, batch in enumerate(self.train_loader) :
            # drop the batch when invalid values 
            if batch is None :
                continue
            input_img, targets, anno_path = batch

            input_img = input_img.to(self.device, non_blocking = True) # 입력이미지도 모델이 위치한 곳에 맞게 설정
            output = self.model(input_img)
            
            # get loss between output value(예측한 객체의 정보) and target value(실제 이미지의 GT 값)
            loss, loss_list = self.yololoss.compute_loss(output, targets, yololayer = self.model.yolo_layers)

            # get gradients
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.schedular_multistep.step(self.iter) # learing rate 조절
            self.iter +=1

            # [loss.item(), lobj.item(), lcls.item(), lbox.item()]
            loss_name = ["total_loss", "obj_loss", "cls_loss", "box_loss"]

            if i % 10 == 0:
                print("epoch {} / iter {} lr {} loss {}".format(self.epoch, self.iter, get_lr(self.optimizer), loss.item()))
                self.torch_writer.add_scalar("lr", get_lr(self.optimizer), self.iter)
                self.torch_writer.add_scalar("total_loss", loss, self.iter)
                for ln, lv in zip(loss_name, loss_list):
                    self.torch_writer.add_scalar(ln, lv, self.iter)

        return loss
                    
    
    def run(self) : # epoch 단위의 for loop
        while True :
            self.model.train()
            # loss calculation   
            loss = self.run_iter()
            self.epoch += 1 # 전체 DB를 한번 다 돌았을 경우 epoch이 1 증가

            # save model (checkpoint)
            checkpoint_path = os.path.join("./output", "model_epoch"+str(self.epoch)+".pth")
            torch.save({"epoch" : self.epoch,
                       "iteration" : self.iter,
                       "model_state_dict" : self.model.state_dict(),
                       "optimizer_state_dict" : self.optimizer.state_dict(),
                       "loss" : loss}, checkpoint_path)

            # evaluation
