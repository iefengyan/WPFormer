import numpy as np
import torch
from torchvision import transforms
import os
from sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from time import time
from PIL import Image
from skimage import io
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.WPFormer import WPFormer
import shutil
import logging
import cv2
from torch.utils.data import Dataset, DataLoader
from mmcv.cnn import get_model_complexity_info


def eval_psnr(test_image_root, test_gt_root, train_size,model):
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    img_transform = transforms.Compose([
        transforms.Resize((train_size,train_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    images = [test_image_root + f for f in os.listdir(test_image_root)]
    gts = [test_gt_root + p for p in os.listdir(test_gt_root)]
    images = sorted(images)
    gts = sorted(gts)


    model.eval()


    for i_test in range(len(images)):
        ori_image=Image.open(images[i_test]).convert("RGB")
        image = img_transform(ori_image).unsqueeze(0).cuda()
        gt = cv2.imread(gts[i_test], cv2.IMREAD_GRAYSCALE)
        H, W = gt.shape
        #
        with torch.no_grad():
                pred = model(image)
                res = pred[-1]
                res = torch.sigmoid(res).data.cpu().numpy().squeeze()


        pred = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pred = Image.fromarray(pred * 255).convert("L")
        pred = pred.resize((W, H), resample=Image.BILINEAR)



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


    results = {
        "MAE": '%.4f' % mae,
        "wFmeasure": '%.4f' % wfm,
        "Smeasure": '%.4f' % sm,
        "meanFm": '%.4f' %fm["curve"].mean(),
        "meanEm": '%.4f' % em["curve"].mean(),
    }
    print(results)

def main(dataset_name):


    net = WPFormer()
    train_size = 384

    if torch.cuda.is_available():
        net=net.cuda()


    model_save = os.path.join("D:\yanfeng\Paper Code\CVPR2025\WPFormer\save",dataset_name+".pth")
    net.load_state_dict(torch.load(model_save),strict=False)


    file_dir = ".\datasets\\"
    test_image_root = os.path.join(file_dir, dataset_name + "\\test\\images\\")
    test_gt_root = os.path.join(file_dir, dataset_name + "\\test\\gt\\")



    eval_psnr( test_image_root, test_gt_root, train_size, net)



if __name__ == '__main__':


    dataset_names = ["ZJU-Leaper","ESDIs-SOD","CrackSeg9k"]
    for dataset_name in dataset_names:

            main(dataset_name)






