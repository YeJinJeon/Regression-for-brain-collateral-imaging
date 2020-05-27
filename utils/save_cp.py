import torch
import os

def save_checkpoint(checkpoint_dir, net, epoch, loss):

    checkpoint_dir = checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_dir = os.path.join(checkpoint_dir,  "models_epoch_%04d_loss_%.10f.pth"%(epoch, loss))

    '''
    if torch.cuda.device_count() > 1:
        state = {'epoch': epoch, 'net': net.module, 'optimizer': optimizer.state_dict()}
    else :
    '''
    state = {'epoch': epoch, 'net': net.module}
    torch.save(state, checkpoint_dir)
    print("Checkpoint saved to {}".format(checkpoint_dir))