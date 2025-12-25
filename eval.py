
import os
import cv2

from sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

def test(test_pred_list,test_gt_list):



    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()

    for i_test in range(len(test_pred_list)):
        # print(i_test)
        gt = cv2.imread(test_gt_list[i_test],cv2.IMREAD_GRAYSCALE)

        h,w = gt.shape
        pred =cv2.imread(test_pred_list[i_test],cv2.IMREAD_GRAYSCALE)
        # if pred.shape != gt.shape:
        #     pred = cv2.resize(pred,dsize=(w,h))
        # print(pred)


        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        M.step(pred=pred, gt=gt)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    mae = M.get_results()["mae"]
    em=EM.get_results()["em"]



    return mae,wfm,sm,fm,em



if __name__ == "__main__":



    datasetnames = ["crackseg9k"]
    methods = ["jtfn","sinetv2","mask2former","bbnet","puenet","FPNet","menet","fspnet","feder","mscafnet","a3net","idenet","zoomnext","FSEL",
                   "camodiffusion","emcadnet","pem"]
    for method in methods :
        for datasetname in datasetnames:
            pred_dir = "D:\yanfeng\project\save\preds"

            test_pred_dir = os.path.join(pred_dir,  method + "\\"+datasetname + "\\")
            # print(test_pred_dir)

            test_gt_dir = os.path.join("F:\SOD_Evaluation_Metrics-main\gt\\", datasetname)
            test_gt_list = [os.path.join(test_gt_dir, file) for file in os.listdir(test_gt_dir)]

            test_pred_list = [os.path.join(test_pred_dir, file) for file in os.listdir(test_pred_dir)]
            test_gt_list = sorted(test_gt_list)

            test_pred_list = sorted(test_pred_list)


            mae,wfm,sm,fm,em= test(test_pred_list,test_gt_list)
            curr_results = {
                "model": method,
                "dataset":datasetname,

                # "MAE": '%.4f' % mae,
                # "Smeasure": '%.4f' % sm,
                # "wFmeasure": '%.4f' % wfm,
                #
                # # E-measure for sod
                # "adpEm": '%.4f' % em["adp"],
                # "meanEm": '%.4f' % em["curve"].mean(),
                # "maxEm": '%.4f' % em["curve"].max(),
                # # F-measure for sod
                # "adpFm": '%.4f' % fm["adp"],
                # "meanFm": '%.4f' % fm["curve"].mean(),
                # "maxFm": '%.4f' % fm["curve"].max(),



                "Smeasure": '%.4f' % sm,
                "wFmeasure": '%.4f' % wfm,
                "meanFm": '%.4f' % fm["curve"].mean(),
                "meanEm": '%.4f' % em["curve"].mean(),
                "MAE": '%.4f' % mae,
            }
            print(curr_results)


