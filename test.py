import time
import  matplotlib.pyplot as plt
from matplotlib import colors
import os
import numpy as np
import statistics as sta

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataload import BrainDataset
from utils.load_model import load_for_test
from unet.model_part import DenseNet
from config import Config
from metrics import *
from utils.evaluation_metrics import RegressionEvaluationMetrics

data_dir = '/media/data1/jeon/workspace/'
checkpoint_dir = '/media/data1/jeon/checkpoints/'
log_dir = '/media/data1/jeon/logs/'
result_dir = '/media/data1/jeon/result/'

def test(opt):

    print()
    print("======TEST=====")
    start_time = time.time()

    metrics = RegressionEvaluationMetrics()

    # load dataset for train
    dataset = BrainDataset(data_dir, 2)  # test:2
    data_loader = DataLoader(dataset, batch_size=1, shuffle= False, num_workers=1)

    # load network
    net = DenseNet(in_ch=60, growth_rate=4, down_theta=0.5)
    _, net= load_for_test(checkpoint_dir, net)
    net = torch.nn.DataParallel(net).to(opt.device)

    # make directory to save result
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    phase = ['ART', 'CAP', 'EVen', 'LVen', 'DEL']
    r_square = {'ART':[], 'CAP':[], 'EVen':[], 'LVen':[], 'DEL':[]}
    mae = {'ART':[], 'CAP':[], 'EVen':[], 'LVen':[], 'DEL':[]}
    tm = {'ART':[], 'CAP':[], 'EVen':[], 'LVen':[], 'DEL':[]}
    ssim = {'ART':[], 'CAP':[], 'EVen':[], 'LVen':[], 'DEL':[]}

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            imgs = batch['image']
            gts = batch['gt']
            masks = batch['binary_mask']

            imgpaths = batch['img_path']
            img_path_list = imgpaths[0].split('/')[5:]
            img_path = '/'.join(img_path_list)

            imgs = imgs.to(opt.device).float()
            gts = gts.to(opt.device).float()
            masks = masks.to(opt.device).float()

            # forward
            out = net(imgs)

            #test with masked out
            out_with_mask = out * masks
            out = out_with_mask.squeeze(0)  # [5, 20, 224, 224]
            gts = gts.squeeze(0)
            masks = masks.squeeze(0)

            channel, depth, width, height = out.shape
            for c in range(channel):
                #each channel
                pred_c = out[c, :, :, :]
                gt_c = gts[c, :, :, :]

                #calculate metrics
                r_square[phase[c]].append(metrics.r_squared(pred_c, gt_c).item())
                mae[phase[c]].append(metrics.mae(pred_c, gt_c))
                tm[phase[c]].append(metrics.TM(pred_c, gt_c).item())
                ssim[phase[c]].append(metrics.ssim(pred_c, gt_c).item())

                #save plot prediction and gt to compare
                #print(plt.rcParams["image.cmap"]) #viridis

                plt.rcParams['figure.figsize'] = [10, 6]
                fig, axes = plt.subplots(2, 5)
                fig.suptitle(phase[c])
                for j in range(5):
                    predict = pred_c[4*j, :, :]
                    gt = gt_c[4*j, :, :]
                    axes[0, j].imshow(predict.cpu()) #prediction
                    axes[1, j].imshow(gt.cpu()) #ground truth
                    axes[0, j].axis('off')
                    axes[1, j].axis('off')
                    axes[0, j].set_title("Prediction")
                    axes[1, j].set_title("GroundTruth")
                #plt.tight_layout()
                save_dir = result_dir+phase[c]+'/'
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                plt.savefig(save_dir+phase[c]+str(i)+'.png')
                #plt.show()
                plt.close()

            print("***\nTesting %d :: %.2fs" % (i, time.time() - start_time))
            print(imgpaths,"\n")

        #print average regression metrics for each phase
        for n, p in enumerate(phase):
            print()
            print(phase[n])
            print("[  Mean  ] r_squared: %.5f, MAE: %.5f, tm: %.5f, ssim: %.5f" %(sum(r_square[phase[n]])/i, sum(mae[phase[n]])/i, sum(tm[phase[n]])/i, sum(ssim[phase[n]])/i))
            print("[ Median ] r_squared: %.5f, MAE: %.5f, tm: %.5f, ssim: %.5f" % (np.median(r_square[phase[n]]), np.median(mae[phase[n]]), np.median(tm[phase[n]]), np.median(ssim[phase[n]])))
            print("[   SD   ] r_squared: %.5f, MAE: %.5f, tm: %.5f, ssim: %.5f" % (np.std(r_square[phase[c]]), np.std(mae[phase[c]]), np.std(tm[phase[c]]), np.std(ssim[phase[c]])))
            print("[  MIN   ] r_squared: %.5f[%d], MAE: %.5f[%d], tm: %.5f[%d], ssim: %.5f[%d]"
                  % (min(r_square[phase[n]]), np.argmin(r_square[phase[n]]), min(mae[phase[n]]), np.argmin(mae[phase[n]]), min(tm[phase[n]]), np.argmin(tm[phase[n]]), min(ssim[phase[n]]), np.argmin(ssim[phase[n]])))
            print("[  MAX   ] r_squared: %.5f[%d], MAE: %.5f[%d], tm: %.5f[%d], ssim: %.5f[%d]"
                  % (max(r_square[phase[n]]), np.argmax(r_square[phase[n]]), max(mae[phase[n]]), np.argmax(mae[phase[n]]), max(tm[phase[n]]), np.argmax(tm[phase[n]]), max(ssim[phase[n]]), np.argmax(ssim[phase[n]])))

        #draw boxplot
        metrics = ["R_squared", "MAE", "TM", "SSIM"]
        for z, m in enumerate([r_square, mae, tm, ssim]):
            #create data for boxplot
            data = m
            data_group = [data[phase[0]], data[phase[1]], data[phase[2]], data[phase[3]]]
            #plot boxplot
            fig, ax = plt.subplots()
            ax.boxplot(data_group, sym="bo", labels=[phase[0], phase[1], phase[2], phase[3]])
            plt.title(metrics[z])
            fig.savefig('/media/data1/jeon/result/bp_'+list(data.keys())[z]+'.png')
            plt.show()
            plt.close()





if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    opt = Config()
    test(opt)
