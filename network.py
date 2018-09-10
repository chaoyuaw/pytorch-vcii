import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvLSTMCell, Sign


class EncoderCell(nn.Module):
    def __init__(self, v_compress, stack, fuse_encoder, fuse_level):
        super(EncoderCell, self).__init__()

        # Init.
        self.v_compress = v_compress
        self.fuse_encoder = fuse_encoder
        self.fuse_level = fuse_level
        if fuse_encoder:
            print('\tEncoder fuse level: {}'.format(self.fuse_level))

        # Layers.
        self.conv = nn.Conv2d(
            9 if stack else 3, 
            64, 
            kernel_size=3, stride=2, padding=1, bias=False)

        self.rnn1 = ConvLSTMCell(
            128 if fuse_encoder and v_compress else 64,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn2 = ConvLSTMCell(
            ((384 if fuse_encoder and v_compress else 256) 
             if self.fuse_level >= 2 else 256),
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn3 = ConvLSTMCell(
            ((768 if fuse_encoder and v_compress else 512) 
             if self.fuse_level >= 3 else 512),
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)


    def forward(self, input, hidden1, hidden2, hidden3,
                unet_output1, unet_output2):

        x = self.conv(input)
        # Fuse
        if self.v_compress and self.fuse_encoder:
            x = torch.cat([x, unet_output1[2], unet_output2[2]], dim=1)

        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]
        # Fuse.
        if self.v_compress and self.fuse_encoder and self.fuse_level >= 2:
            x = torch.cat([x, unet_output1[1], unet_output2[1]], dim=1)

        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]
        # Fuse.
        if self.v_compress and self.fuse_encoder and self.fuse_level >= 3:
            x = torch.cat([x, unet_output1[0], unet_output2[0]], dim=1)

        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]
        return x, hidden1, hidden2, hidden3


class Binarizer(nn.Module):
    def __init__(self, bits):
        super(Binarizer, self).__init__()
        self.conv = nn.Conv2d(512, bits, kernel_size=1, bias=False)
        self.sign = Sign()

    def forward(self, input):
        feat = self.conv(input)
        x = F.tanh(feat)
        return self.sign(x)


class DecoderCell(nn.Module):
    def __init__(self, v_compress, shrink, bits, fuse_level):

        super(DecoderCell, self).__init__()

        # Init.
        self.v_compress = v_compress
        self.fuse_level = fuse_level
        print('\tDecoder fuse level: {}'.format(self.fuse_level))

        # Layers.
        self.conv1 = nn.Conv2d(
            bits, 512, kernel_size=1, stride=1, padding=0, bias=False)

        self.rnn1 = ConvLSTMCell(
            512,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn2 = ConvLSTMCell(
            (((128 + 256 // shrink * 2) if v_compress else 128) 
             if self.fuse_level >= 3 else 128), #out1=256
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn3 = ConvLSTMCell(
            (((128 + 128//shrink*2) if v_compress else 128) 
             if self.fuse_level >= 2 else 128), #out2=128
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)

        self.rnn4 = ConvLSTMCell(
            (64 + 64//shrink*2) if v_compress else 64, #out3=64
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)

        self.conv2 = nn.Conv2d(
            32,
            3, 
            kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, input, hidden1, hidden2, hidden3, hidden4,
                unet_output1, unet_output2):

        x = self.conv1(input)
        hidden1 = self.rnn1(x, hidden1)

        # rnn 2
        x = hidden1[0]
        x = F.pixel_shuffle(x, 2)

        if self.v_compress and self.fuse_level >= 3:
            x = torch.cat([x, unet_output1[0], unet_output2[0]], dim=1)

        hidden2 = self.rnn2(x, hidden2)

        # rnn 3
        x = hidden2[0]
        x = F.pixel_shuffle(x, 2)

        if self.v_compress and self.fuse_level >= 2:
            x = torch.cat([x, unet_output1[1], unet_output2[1]], dim=1)

        hidden3 = self.rnn3(x, hidden3)

        # rnn 4
        x = hidden3[0]
        x = F.pixel_shuffle(x, 2)

        if self.v_compress:
            x = torch.cat([x, unet_output1[2], unet_output2[2]], dim=1)

        hidden4 = self.rnn4(x, hidden4)

        # final
        x = hidden4[0]
        x = F.pixel_shuffle(x, 2)

        x = F.tanh(self.conv2(x)) / 2
        return x, hidden1, hidden2, hidden3, hidden4

