import torch
import torch.nn as nn 

from utils.tools import *
import os, sys

class Yololoss(nn.Module):
    def __init__(self, device, num_class):
        super(Yololoss, self). __init__()
        self.device = device
        self.num_class = num_class
        self.mseloss = nn.MSELoss().to(device)
        self.bceloss = nn.BCELoss().to(device)
        self.bcelogloss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([1.0], device = device)).to(device)

    def compute_loss(self, pred, targets=None, yololayer=None) : # yololayer은 각각 실제 모델의 yololayer을 가져오는 역할을 함

        # loss는 3가지 종류 (class(예측한 해당 클래스가 맞는지?), bbox(박스가 정확하게 잡고있는지?), objectness(이게 object인지 아닌지?))
        lcls, lbox, lobj = torch.zeros(1, device = self.device), torch.zeros(1, device = self.device), torch.zeros(1, device = self.device) # 초기값을 0으로 설정

        # get positive targets
        tcls, tbox, tindicies, tanchors = self.get_targets(pred, targets, yololayer)

        # 3 yolo layers
        for pidx, pout in enumerate(pred) : 
            batch_id, anchor_id, gy, gx = tindicies[pidx]

            tobj = torch.zeros_like(pout[...,0], device = self.device)
            
            num_targets = batch_id.shape[0]

            if num_targets:
                ps = pout[batch_id, anchor_id, gy, gx]
                pxy = torch.sigmoid(ps[..., 0:2])
                pwh = torch.exp(ps[..., 2:4]) * tanchors[pidx]
                pbox = torch.cat((pxy, pwh),1)
                iou = bbox_iou(pbox.T, tbox[pidx], xyxy = False)

                # box loss
                # MSE(Mean Squared Loss)
                # loss_wh = self.mseloss(pbox[...,2:4], tbox[pidx][...,2:4])
                # loss_xy = self.mseloss(pbox[...,0:2], tbox[pidx][...,0:2])
                lbox += (1 - iou).mean()                

                # objectness
                # gt box와 predicted box가 겹치면 positive : 1, 안 겹치면 negative : 0 => using IOU
                tobj[batch_id, anchor_id, gy, gx] = iou.detach().clamp(0).type(tobj.dtype)
                
                # class loss
                if ps.size(1) - 5 > 1 :
                    t = torch.zeros_like(ps[...,5:], device=self.device)
                    t[range(num_targets), tcls[pidx]] =1 
                    lcls += self.bcelogloss(ps[:,5:], t)
                
            lobj += self.bcelogloss(pout[...,4], tobj)
        
        # loss weight
        lcls *= 0.05
        lobj *= 1.0
        lbox *= 0.5

        # total loss
        loss = lcls + lbox + lobj
        loss_list = [loss.item(), lobj.item(), lcls.item(), lbox.item()]
        
        return loss, loss_list

        '''
        pout.shape = [batch, anchors, grid_y, grid_x, box_attrib]  / 19 x 19, 38 x 38, 76 x 76 
        the number of boxes in each yolo layer : anchors * grid_y * grid_w
        yolo 0 -> 3 * 19 * 19, yolo 1 -> 3 * 38 * 38, yolo2 -> 3 * 76 * 76
        1개의 이미지에서 나오는 total boxes : 22743
        
        positive prediction(gt와 예측한 박스가 어느 정도 겹치고 실제 긍정적인 값, 잘 예측했다고 생각되는 값) vs negative prediction(그냥 백그라운드를 예측하는 값)
        pos : neg = 0.01 : 0.99 정도로 심하게 차이가 남, pos 가 많아봐야 30개정도가 된다면 22743- 30 개 정도가 됨 
        -> 유의미한 postivie prediction을 잘 계산해야하는데 negative가 많으면 원하는 결과를 못 도출함 

        only positive prediction에서만 box_loss와 class_loss를 뽑아낼 수 있음 (gt 값이 있는 부분만 비교가 가능하므로)
        -> negative prediction에서는 only obj_loss만을 뽑아낼 수 있음 
        '''
    
    def get_targets(self, preds, targets, yololayer) :
  
        num_anc = 3
        num_targets = targets.shape[0]
        tcls, tboxes, indices, anch = [], [], [], [] # target 클래스, target 박스, 해당 인덱스 정보, 해당 anchor 정보

        gain = torch.ones(7, device = self.device, dtype=torch.int64) # target의 값에 anchor을 추가하고 싶음

        # anchor_index
        ai = torch.arange(num_anc, device = targets.device).float().view(num_anc, 1).repeat(1, num_targets) # (num_anc,1)로 transpose 후 num_targets 만큼 반복
                
        # targets shape : [batch_id, class_id, box_cx, box_cy, box_w, box_h, anchor_id]
        targets = torch.cat((targets.repeat(num_anc,1,1), ai[:, :, None]), 2).to(self.device)
   
        for yi, yl in enumerate(yololayer):
            anchors = yl.anchor / yl.stride # yolo layer의 scale에 맞게 각 anchor을 만듬

            gain[2:6] = torch.tensor(preds[yi].shape)[[3, 2, 3, 2]] # grid_w, grid_h
            
            t = targets * gain # normalize 된 값들을 다시 yolo_layer의 grid shape에 맞게 resolution함

            if num_targets:
                r = t[:, :, 4:6] / anchors[:, None] # target값의 w, h 값에 anchor을 나눔

                j = torch.max(r, 1. / r).max(2)[0] < 4

                t = t[j]

            else : 
                t = targets[0]

            # batch_id, class_id
            b, c = t[:, :2].long().T

            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            
            gij = gxy.long()
            
            gi, gj = gij.T
            
            #anchor index
            a = t[:, 6].long()
            
            #add index list
            indices.append((b, a, gj.clamp_(0,gain[3]-1), gi.clamp_(0,gain[2]-1)))     
            
            #add target box
            tboxes.append(torch.cat((gxy-gij, gwh),1))
            
            #add anchor
            anch.append(anchors[a])
            
            #add class of each target
            tcls.append(c)

        return tcls, tboxes, indices, anch
     
    