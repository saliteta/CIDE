# 100 -> 4*4*1024 -> 8*8*512 -> 16*16*256 -> 32*32*128 -> 64*64*3
# 64*64 rgb image
# exact same but in a reverse order for descriminator
'''
• Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
• Use batchnorm in both the generator and the discriminator.
• Remove fully connected hidden layers for deeper architectures.
• Use ReLU activation in generator for all layers except for the output, which uses Tanh.
• Use LeakyReLU activation in the discriminator for all layers.
'''

from turtle import forward
import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, channel_img, feature_d) -> None:
        super().__init__()
        # input is 64*64
        self.disc = nn.Sequential(
            nn.Conv2d(
                channel_img,
                feature_d,
                kernel_size=4,
                stride=2,
                padding=1
            ),
        # (64-4+2)/2+1 = 32 (width-kernal size + 2*padding)/2 + 1
        # inverse order
            nn.LeakyReLU(0.2),
            self._block(feature_d, feature_d*2,4,2,1), # 16*16*256
            self._block(feature_d*2, feature_d*4,4,2,1), # 8*8*512
            self._block(feature_d*4, feature_d*8,4,2,1), #4*4*1024
            nn.Conv2d(
                feature_d*8, 1, kernel_size=4, stride= 2, padding=0
            ),
            # 1*1, defualt out channel is 1
        )
    
    def _block(self, in_channel, out_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride,
                padding,
                bias = False,#use batch norm
            ),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self,x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, channel_img, features_g) -> None:#z means noise
        super().__init__()
        self.gen = nn.Sequential(
            # one may wonder why the input is z_dim 
            self._block(z_dim, features_g*16,4,1,0), # don't know the length of the noise
            self._block(features_g*16, features_g*8,4,2,1), # the length is 1024/2 if the feature_g is 64
            self._block(features_g*8, features_g*4,4,2,1), 
            self._block(features_g*4, features_g*2,4,2,1), 

            nn.ConvTranspose2d(
                features_g*2, channel_img, kernel_size=4, stride = 2, padding=1,
            ),
            nn.Tanh(),
        )
    
    def _block(self, in_channel, out_channel, kernel_size, stride, padding):
        #up scale
        return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    stride,
                    padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
        )
    def forward(self,x):
        return self.gen(x)

def initialize_weight(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

