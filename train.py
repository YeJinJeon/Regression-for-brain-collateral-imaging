import os
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from unet.model_part import DenseNet
from dataload import BrainDataset
from config import Config
from utils.save_cp import save_checkpoint
from utils.load_model import load_model_best
from phase_loss import average_phase_loss

data_dir = '/media/data1/jeon/workspace/'
checkpoint_dir = '/media/data1/jeon/checkpoints_lr/'
log_dir = '/media/data1/jeon/logs_lr/'


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer(Config):

    def train(self, net, optimizer, scheduler, epoch, writer):
        print("=======Training======")
        start_time = time.time()
        total_loss =0.0  #for average loss per epoch

        #load dataset for train
        dataset = BrainDataset(data_dir, 0) #train:0
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

        for i, batch in enumerate(data_loader):
            imgpaths = batch['img_path']
            imgs = batch['image']
            gts = batch['gt']
            weight_masks = batch['weight_mask']
            masks = batch['binary_mask']

            imgs = imgs.to(self.device).float()
            gts = gts.to(self.device).float()
            weight_masks = weight_masks.to(self.device).float()
            masks = masks.to(self.device).float()

            #forward
            t0 = time.time()
            out = net(imgs)

            #Loss: weighted mse(mean squred error) or L2 norm
            loss = average_phase_loss(out, gts, weight_masks, masks)
            print(loss)
            total_loss += loss.item()

            # gradient update
            optimizer.zero_grad() #역전파 단계를 실행하기 전에 변화도: 0, optimizer 정의시 net parameter 포함
            loss.backward()
            optimizer.step()
            # scheduler.step()
            t1 = time.time()

            #print every 10 batch
            if i % 10 == 0:
                print('timer: %.4f sec. ' % (t1-t0))
                print('iter' + str(i) + " || Loss: %.4f" % (loss), end=' ')

        #averge loss per epoch
        total_loss = total_loss / i
        #print every epoch
        print("***\nTraining %.2fs => Epoch[%d/%d] :: Loss : %.10f\n" % (time.time() - start_time, epoch, self.n_epochs, total_loss))

        return total_loss

    def infer(self, net, epoch, writer):

        print('====Validation=====')
        start_time = time.time()
        total_loss = 0.0

        dataset = BrainDataset(data_dir, 1)  # val:1
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                imgs = batch['image']
                gts = batch['gt']
                weight_masks = batch['weight_mask']
                masks = batch['binary_mask']

                imgs = imgs.to(self.device).float()
                gts = gts.to(self.device).float()
                weight_masks = weight_masks.to(self.device).float()
                masks = masks.to(self.device).float()

                #forward
                out = net(imgs)

                # loss
                loss = average_phase_loss(out, gts, weight_masks, masks)

                total_loss += loss.item()

            # averge loss
            total_loss = total_loss / i
            print("***\nValidation %.2fs => Epoch[%d/%d] :: Loss : %.10f\n" % (time.time() - start_time, epoch, self.n_epochs, total_loss))

            return total_loss


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    #seed_everything(1234)
    #클래스 생성자
    opt = Config()
    trainer = Trainer()
    start_epoch = 0

    # trainer.seed_everything()

    # load network
    net = DenseNet(in_ch=60, growth_rate=4, down_theta=0.5)

    # optimizers
    optimizer = optim.Adam(net.parameters(), lr=opt.init_lr, betas=(opt.b1, opt.b2), eps=1e-08)

    # resume train
    if opt.resume == True:
        start_epoch, net = load_model_best(checkpoint_dir, net)

    net = torch.nn.DataParallel(net).to(opt.device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, opt.lr_steps, gamma=0.9)

    # define writer
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    for epoch in range(start_epoch, opt.n_epochs):

        #scheduler.step()
        #lr = scheduler.get_lr()

        train_loss = trainer.train(net, optimizer, scheduler, epoch, writer)
        valid_loss = trainer.infer(net, epoch, writer)

        #write summary
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)

        # save checkpoint per epoch
        save_checkpoint(checkpoint_dir, net, epoch, valid_loss)

    writer.close()



