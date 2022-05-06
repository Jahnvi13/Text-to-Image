import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import Concat_embed
import pdb
import torch.nn.functional as F
from collections import OrderedDict

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.noise_dim = 100
        self.embed_dim = 512
        self.projected_embed_dim = 128
        self.latent_dim = self.noise_dim + self.projected_embed_dim
        self.ngf = 64

        self.projection = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
            nn.BatchNorm1d(num_features=self.projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        # Affine blocks do not change the dimensions of the input

        # self.affine0 = affine(self.ngf * 16)
        # self.affine1 = affine(self.ngf * 8)
        # self.affine2 = affine(self.ngf * 4)
        self.affine3 = affine(self.ngf * 2)
        self.affine4 = affine(self.ngf * 1)

        # based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
        # First block increases h and w dimensions to 4 times, rest double.
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 2,self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, embed_vector, z):

        projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
        latent_vector = torch.cat([projected_embed, z], 1)

        output = self.conv1(latent_vector)
        # output = self.affine1(output,projected_embed.squeeze())
        # print(output.shape)
        output = F.interpolate(output,size=(output.shape[2]*2,output.shape[3]*2),mode='nearest')
        # print(output.shape)
        output = self.conv2(output)
        # output = self.affine2(output,projected_embed.squeeze())
        # print(output.shape)
        output = self.conv3(output)
        output = self.affine3(output,projected_embed.squeeze())
        # print(output.shape)
        output = self.conv4(output)
        output = self.affine4(output,projected_embed.squeeze())
        # print(output.shape)
        output = self.conv5(output)
        # print(output.shape)
        
        return output


class affine(nn.Module):

    def __init__(self, num_features):
        super(affine, self).__init__()
        # self.y=y
        # print("affine",num_features)
        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(128, 128)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(128, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(128, 128)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(128, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
        # y=self.y
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)        

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.embed_dim = 512
        self.projected_embed_dim = 128
        self.ndf = 64
        self.B_dim = 128
        self.C_dim = 16

        self.netD_1 = nn.Sequential(
        	# each consequent layer reduces h and w dimensions by half
        	nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
        	nn.LeakyReLU(0.2, inplace=True),
        	
        	nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
        	nn.BatchNorm2d(self.ndf * 2),
        	nn.LeakyReLU(0.2, inplace=True),
        	
        	nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
        	nn.BatchNorm2d(self.ndf * 4),
        	nn.LeakyReLU(0.2, inplace=True),
        	
        	nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
        	nn.BatchNorm2d(self.ndf * 8),
        	nn.LeakyReLU(0.2, inplace=True),
        )


        self.projector = Concat_embed(self.embed_dim, self.projected_embed_dim)

        self.netD_2 = nn.Sequential(
            
            nn.Conv2d(self.ndf * 8 + self.projected_embed_dim, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            )

    def forward(self, inp, embed):
        x_intermediate = self.netD_1(inp)

        x_intermediate=F.interpolate(x_intermediate,size=(4,4),mode='nearest')
        
        x = self.projector(x_intermediate, embed)
        x = self.netD_2(x)

        return x.view(-1, 1).squeeze(1) , x_intermediate