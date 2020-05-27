import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.group_norm import GroupNorm3d
import torchsummary

"""
modified
    https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    DenseNet: https://hoya012.github.io/blog/DenseNet-Tutorial-2/
"""

'''
class Net(nn.Module):
    """
    A base class provides a common weight initialization scheme.
    """
    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            if 'conv' in classname.lower():
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            if 'norm' in classname.lower():
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if 'linear' in classname.lower():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        return x
'''

class FirstConv(nn.Module):
    """(Conv3d - Group Normalization - ReLU)"""

    def __init__(self, in_ch, out_ch):
        super(FirstConv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=(3,3,3), stride = 1, padding=1)
        self.group_norm = GroupNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool3d = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(1, 2, 2), padding=(1, 0, 0))

    def forward(self, x):
        out = self.conv(x)
        out = self.group_norm(out)
        out = self.relu(out)
        out_pool= self.max_pool3d(out)
        return out, out_pool


class BasicConv(nn.Module):
    """(Group Normaliztion - ReLU - Conv3d)"""

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bias=False):
        super(BasicConv, self).__init__()
        self.group_norm = GroupNorm3d(in_ch)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        out = self.group_norm(x)
        out= self.relu(out)
        out=self.conv(out)
        return out

class GroupConv(nn.Module):
    """(Group Normaliztion - ReLU - Group Conv3d)"""

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, group, bias=False):
        super(GroupConv, self).__init__()
        self.group_norm = GroupNorm3d(in_ch)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=group)

    def forward(self, x):
        out = self.group_norm(x)
        out= self.relu(out)
        out=self.conv(out)
        return out

class LastConv(nn.Module):
    """(Conv3d - Group Normalization - ReLU)"""

    def __init__(self, in_ch, out_ch):
        super(LastConv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=(1,1,1), stride = 1, padding=0)
        self.group_norm = GroupNorm3d(out_ch)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.conv(x)
        out = self.group_norm(out)
        out = self.tanh(out)
        return out

class Dense_layer(nn.Sequential):
    """(BasicConv 1 w/ kernel size (1,1,3) - BasicConv 2 w/ kernel size (3,3,3)"""

    def __init__(self, in_ch, growth_rate):
        super(Dense_layer, self).__init__()
        self.add_module('conv_1x1',
                        BasicConv(in_ch=in_ch, out_ch=growth_rate*4, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False))
        self.add_module('conv_3x3',
                        BasicConv(in_ch=growth_rate*4, out_ch=growth_rate, kernel_size=(3,3,3), stride=(1, 1, 1), padding=1, bias=False))

    def forward(self, x):
        dense_output  = super(Dense_layer, self).forward(x)
        dense_output = torch.cat((x, dense_output), 1)
        return dense_output


class Transition_layer(nn.Sequential):
    """(Conv(1,1,3):1/2 number of feature maps - Maxpool3d:1/2 feature map size)"""

    def __init__(self, in_ch, theta=0.5):
        super(Transition_layer, self).__init__()
        self.add_module('conv_1x1',
                        BasicConv(in_ch=in_ch, out_ch=int(in_ch * theta), kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False))

    def forward(self, x):
        transition_output = super(Transition_layer, self).forward(x)
        return transition_output

class DenseBlock(nn.Sequential):
    """num_dense_layers = [6, 12, 24, 48]"""
    def __init__(self, in_ch, num_dense_layers, growth_rate):
        super(DenseBlock, self).__init__()

        for i in range(num_dense_layers):
            in_ch_dense_layer = in_ch + growth_rate * i
            self.add_module('dense_block_%d' % i,
                            Dense_layer(in_ch = in_ch_dense_layer, growth_rate = growth_rate))

    def forward(self, x):
        dense_block_output  = super(DenseBlock, self).forward(x)
        return dense_block_output


class conv3d_trans_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv3d_trans_block, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0))
    def forward(self, x):
        conv3d_trans_output = self.conv3d_transpose(x)
        return conv3d_trans_output


class ResidualBlock(nn.Module):
    def __init__(self, in_ch):
        super(ResidualBlock, self).__init__()

        self.residual_block = nn.Sequential(
            BasicConv(in_ch=in_ch, out_ch=in_ch, kernel_size=(3, 1, 1), stride=1, padding=(1,0,0), bias=False),
            GroupConv(in_ch=in_ch, out_ch=in_ch, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False, group=8),
            BasicConv(in_ch=in_ch, out_ch=in_ch, kernel_size=(3, 1, 1), stride=1, padding=(1,0,0), bias=False)
        )
    def forward(self, x):
        residual_block_output = self.residual_block(x)
        return residual_block_output


class DenseNet(nn.Module):
    def __init__(self, in_ch=60, growth_rate=4, down_theta=0.5, up_theta=2):
        """down_theta = 1/2 * number of feature maps in transition layer
            up_theta = 2 * number of feature maps in transition layer"""
        super(DenseNet, self).__init__()

        init_out_ch = 8
        num_dense_layers = [6, 12, 24, 48]

        self.First_layer = FirstConv(in_ch, init_out_ch)

        self.dense_block_1 = DenseBlock(init_out_ch, num_dense_layers=num_dense_layers[0], growth_rate=growth_rate)
        in_ch_transition_layer_1 = init_out_ch + (growth_rate * num_dense_layers[0])
        self.transition_layer_1 = Transition_layer(in_ch=in_ch_transition_layer_1, theta=down_theta)

        self.max_pool3d_2 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(1, 2, 2), padding=(1, 0, 0)) #1/2 feature map width, height
        self.dense_block_2 = DenseBlock(int(in_ch_transition_layer_1*0.5), num_dense_layers=num_dense_layers[1], growth_rate=growth_rate)
        in_ch_transition_layer_2 = int(in_ch_transition_layer_1*0.5) + (growth_rate * num_dense_layers[1])
        self.transition_layer_2 = Transition_layer(in_ch=in_ch_transition_layer_2, theta=down_theta)

        self.max_pool3d_3 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(1, 2, 2), padding=(1, 0, 0))  # 1/2 feature map width, height
        self.dense_block_3 = DenseBlock(int(in_ch_transition_layer_2*0.5), num_dense_layers=num_dense_layers[2], growth_rate=growth_rate)
        in_ch_transition_layer_3 = int(in_ch_transition_layer_2*0.5) + (growth_rate * num_dense_layers[2])
        self.transition_layer_3 = Transition_layer(in_ch=in_ch_transition_layer_3, theta=down_theta)

        self.max_pool3d_4 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(1, 2, 2), padding=(1, 0, 0))  # 1/2 feature map width, height
        self.dense_block_4 = DenseBlock(int(in_ch_transition_layer_3*0.5), num_dense_layers=num_dense_layers[3], growth_rate=growth_rate)
        in_ch_transition_layer_4 = int(in_ch_transition_layer_3*0.5) + (growth_rate * num_dense_layers[3])
        self.transition_layer_4 = Transition_layer(in_ch=in_ch_transition_layer_4, theta=down_theta)

        ##UP
        in_ch_up_block_1 = int(in_ch_transition_layer_4 * down_theta) #128
        self.con3d_tran_1 = conv3d_trans_block(in_ch=in_ch_up_block_1, out_ch=in_ch_up_block_1)
        self.residual_decode_1 = ResidualBlock(in_ch = in_ch_up_block_1 * up_theta) #256

        in_ch_up_block_2 = in_ch_up_block_1 #128
        out_ch_up_block_2 = int(in_ch_up_block_2 * down_theta) #64
        self.con3d_tran_2 = conv3d_trans_block(in_ch=in_ch_up_block_2, out_ch=out_ch_up_block_2)
        self.residual_decode_2 = ResidualBlock(in_ch=in_ch_up_block_2)

        in_ch_up_block_3 = out_ch_up_block_2 #64
        out_ch_up_block_3 = int(out_ch_up_block_2 * down_theta)  # 32
        self.con3d_tran_3 = conv3d_trans_block(in_ch=in_ch_up_block_3, out_ch=out_ch_up_block_3)
        self.residual_decode_3 = ResidualBlock(in_ch=in_ch_up_block_3)

        in_ch_up_block_4 = out_ch_up_block_3 #32
        out_ch_up_block_4 = int(out_ch_up_block_3 * down_theta * down_theta) #8
        self.con3d_tran_4 = conv3d_trans_block(in_ch=in_ch_up_block_4, out_ch=out_ch_up_block_4)
        self.residual_decode_4 = ResidualBlock(in_ch=int(in_ch_up_block_4 * down_theta))

        in_ch_last_conv = in_ch_up_block_4
        self.Lastlayer = LastConv(in_ch=in_ch_last_conv, out_ch=5)

    def forward(self, x):

        ##DOWN

        init_layer_output, init_layer_output_pool = self.First_layer(x)
        #print("First layer output : ", init_layer_output_pool.shape)

        dense_block_1_output = self.dense_block_1(init_layer_output_pool)
        transition_layer_1_output = self.transition_layer_1(dense_block_1_output)
        #print("Dense Block 1 : ", dense_block_1_output.shape)
        #print("Transition Block 1: ", transition_layer_1_output.shape)

        dense_block_2_input = self.max_pool3d_2(transition_layer_1_output)
        dense_block_2_output = self.dense_block_2(dense_block_2_input)
        transition_layer_2_output = self.transition_layer_2(dense_block_2_output)
        #print("Dense Block 2 : ", dense_block_2_output.shape)
        #print("Transition Block 2: ", transition_layer_2_output.shape)

        dense_block_3_input = self.max_pool3d_3(transition_layer_2_output)
        dense_block_3_output = self.dense_block_3(dense_block_3_input)
        transition_layer_3_output = self.transition_layer_3(dense_block_3_output)
        #print("Dense Block 3 : ", dense_block_3_output.shape)
        #print("Transition Block 3: ", transition_layer_3_output.shape)

        dense_block_4_input = self.max_pool3d_4(transition_layer_3_output)
        dense_block_4_output = self.dense_block_4(dense_block_4_input)
        transition_layer_4_output = self.transition_layer_4(dense_block_4_output)
        #print("Dense Block 4 : ", dense_block_4_output.shape)
        #print("Transition Block 4: ", transition_layer_4_output.shape)

        ##UP
        #print("==========UP===========")
        conv3d_trans_block_output_1 = self.con3d_tran_1(transition_layer_4_output)
        conv3d_trans_block_concat_dense_1 = torch.cat((conv3d_trans_block_output_1, dense_block_3_output),1)
        residual_decode_output_1 = self.residual_decode_1(conv3d_trans_block_concat_dense_1)
        up_block_1_output = torch.cat((residual_decode_output_1, conv3d_trans_block_concat_dense_1), 1)
        #print("UP Block 1 output: ", up_block_1_output.shape)

        conv3d_trans_block_output_2 = self.con3d_tran_2(conv3d_trans_block_output_1)
        conv3d_trans_block_concat_dense_2 = torch.cat((conv3d_trans_block_output_2, dense_block_2_output), 1)
        residual_decode_output_2 = self.residual_decode_2(conv3d_trans_block_concat_dense_2)
        up_block_2_output = torch.cat((residual_decode_output_2, conv3d_trans_block_concat_dense_2), 1)
        #print("UP Block 2 output: ", up_block_2_output.shape)

        conv3d_trans_block_output_3 = self.con3d_tran_3(conv3d_trans_block_output_2)
        conv3d_trans_block_concat_dense_3 = torch.cat((conv3d_trans_block_output_3, dense_block_1_output), 1)
        residual_decode_output_3 = self.residual_decode_3(conv3d_trans_block_concat_dense_3)
        up_block_3_output = torch.cat((residual_decode_output_3, conv3d_trans_block_concat_dense_3), 1)
        #print("UP Block 3 output: ", up_block_3_output.shape)

        conv3d_trans_block_output_4 = self.con3d_tran_4(conv3d_trans_block_output_3)
        conv3d_trans_block_concat_dense_4 = torch.cat((conv3d_trans_block_output_4, init_layer_output), 1)
        residual_decode_output_4 = self.residual_decode_4(conv3d_trans_block_concat_dense_4)
        up_block_4_output = torch.cat((residual_decode_output_4, conv3d_trans_block_concat_dense_4), 1)
        #print("UP Block 4 output: ", up_block_4_output.shape)

        final_output = self.Lastlayer(up_block_4_output)
        #print("final output: ", final_output.shape)

        return  final_output


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    input_test = torch.randn(3, 8, 96, 96, 96) #(batch_size, channels, depth, height, width)
    input_test = input_test.cuda()
    print("The shape of input: ", input_test.shape)

    net = DenseNet(growth_rate=4, down_theta=0.5)
    net.cuda()
    print(net)

    out = net(input_test)
    print("The shape of output: ", out.shape)


