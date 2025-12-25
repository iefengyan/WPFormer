
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

import os


import logging

import sys

from sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from PIL import Image
from skimage import io
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.WPFormer import WPFormer
from sod_dataloader import get_loader
import cv2



def setup_logger(name, save_dir, filename="log.txt", mode='w'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode)  # 'a+' for add, 'w' for overwrite
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger




def eval_psnr(test_image_root, test_gt_root, train_size,model):

    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()

    img_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    images = [test_image_root + f for f in os.listdir(test_image_root)]
    gts = [test_gt_root + p for p in os.listdir(test_gt_root)]
    images = sorted(images)
    gts = sorted(gts)
    model.eval()
    for index in range(len(images)):
        ori_image=Image.open(images[index]).convert("RGB")
        image = img_transform(ori_image).unsqueeze(0).cuda()

        gt = cv2.imread(gts[index], cv2.IMREAD_GRAYSCALE)
        H, W = gt.shape
        with torch.no_grad():
            predictions_mask = model(image)
            res=predictions_mask[-1]
        res = F.interpolate(res, size=(H, W), mode="bilinear", align_corners=False)
        res = torch.sigmoid(res).data.cpu().numpy().squeeze()
        pred = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pred = Image.fromarray(pred * 255).convert("L")

        pred = np.array(pred)



        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        M.step(pred=pred, gt=gt)
        #
    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    curr_results = {
        "MAE": mae,
        "Smeasure": sm,
        "wFmeasure": wfm,
        "meanEm": em["curve"].mean(),
    }


    return mae,wfm

def total_loss(pred, mask):
    pred = torch.sigmoid(pred)
    bce_loss = nn.BCELoss()
    bce = bce_loss(pred, mask)

    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - inter/(union-inter)
    iou = iou.mean()

    #mse_loss = nn.MSELoss(reduction="mean")
    #mse = mse_loss(pred, mask)

    return iou+bce
import time
def train(model_name, dataset_name):


    if dataset_name == "COD":
        epoch_num = 150
        epoch_val = 100
    elif dataset_name == "SOD":
        epoch_num = 60
        epoch_val = 30

    net = WPFormer(channel=128)
    train_size = 384

    file_dir= ".\datasets\\"

    train_image_root = os.path.join(file_dir, dataset_name + "\\train\images\\")
    train_gt_root = os.path.join(file_dir, dataset_name + "\\train\gt\\")
    test_image_root = os.path.join(file_dir, dataset_name + "\\test\\ECSSD\imgs\\")
    test_gt_root = os.path.join(file_dir, dataset_name + "\\test\\ECSSD\gt\\")
    # For COD, the bs is set to 16. For SOD,  the bs is set to 6.
    train_loader1 = get_loader(train_image_root, train_gt_root, batchsize=6, trainsize=train_size, is_train=True)

    # ------- 3. define model --------
    if torch.cuda.is_available():
        net=net.cuda()


    # ------- 4. define optimizer --------
    print("---define optimizer...")

    optimizer = optim.Adam(net.parameters(), lr=4e-5)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------- 5. training process --------
    print("---start training...")

    running_loss = 0.0


    best_sm=0


    for epoch in range(0, epoch_num):
        print(epoch)
        start_time = time.time()

        for i, data in enumerate(train_loader1):


            inputs, labels = data['image'], data['label']
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            images, gts = Variable(inputs.to(device), requires_grad=False), Variable(labels.to(device),
                                                                                     requires_grad=False)
            #


            # y zero the parameter gradients
            optimizer.zero_grad()
            predictions_mask = net(images)

            mask_losses=0

            for i in range(len(predictions_mask)):
                mask_losses = mask_losses + total_loss(predictions_mask[i], gts)

            losses = mask_losses
            losses.backward()
            optimizer.step()

            running_loss += losses.item()


        end_time = time.time()
        print('Cost time: {:.4f}'.format(end_time - start_time))

        lr_scheduler.step()


        if (epoch+1) >=epoch_val:
            mae, sm = eval_psnr(test_image_root, test_gt_root, train_size,net)

            if sm > best_sm:
                save_path=os.path.join(".\save",dataset_name)
                torch.save(net.state_dict(), os.path.join(save_path, model_name+"-"+dataset_name+'-'+f'{sm:.4f}'+'.pth'))
                best_sm = sm
                print(best_sm)
            print("mae:%.4f, best_sm:%.4f, sm: %.4f" % (mae, best_sm, sm))
            net.train()





if __name__ == '__main__':
    # setup_seed(2025)

    # dataset_name = "SOD"
    # train(model_name, dataset_name)
    dataset_name = "SOD"
    train(dataset_name)
  

