import os
import glob
import torch


def load_model_best(checkpoint_dir, net):

    checkpoint_list = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    checkpoint_list.sort()

    n_epoch = 0

    loss_list = list(map(lambda x: float(os.path.basename(x).split('_')[4][:-4]), checkpoint_list))
    best_loss_idx = loss_list.index(min(loss_list))
    checkpoint_path = checkpoint_list[best_loss_idx]

    if os.path.isfile(checkpoint_path):
        # print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)

        n_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'].state_dict())
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, n_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))

    return n_epoch + 1, net

def load_for_test(checkpoint_dir, net):
    checkpoint_list = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    checkpoint_list.sort()

    n_epoch = 0

    loss_list = list(map(lambda x: float(os.path.basename(x).split('_')[4][:-4]), checkpoint_list))
    best_loss_idx = loss_list.index(min(loss_list))
    checkpoint_path = checkpoint_list[best_loss_idx]

    if os.path.isfile(checkpoint_path):
        # print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)

        n_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'].state_dict())
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, n_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))

    return n_epoch + 1, net