import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from CustomDataset import CustomDataset




class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.out_size = (384,512)


        self.relu = nn.ReLU()

        self.conv_enc_1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=0)
        self.bn_conv_enc_1_1= nn.BatchNorm2d(64)
        self.conv_enc_1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.bn_conv_enc_1_2= nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_enc_2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
        self.bn_conv_enc_2_1 = nn.BatchNorm2d(128)
        self.conv_enc_2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=0)
        self.bn_conv_enc_2_2 = nn.BatchNorm2d(128)

        self.conv_enc_3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=0)
        self.bn_conv_enc_3_1 = nn.BatchNorm2d(256)
        self.conv_enc_3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=0)
        self.bn_conv_enc_3_2 = nn.BatchNorm2d(256)

        self.conv_enc_4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=0)
        self.bn_conv_enc_4_1 = nn.BatchNorm2d(512)
        self.conv_enc_4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=0)
        self.bn_conv_enc_4_2 = nn.BatchNorm2d(512)

        self.conv_enc_5_1 = nn.Conv2d(512, 1024, kernel_size=3, padding=0)
        self.bn_conv_enc_5_1 = nn.BatchNorm2d(1024)
        self.conv_enc_5_2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=0)
        self.bn_conv_enc_5_2 = nn.BatchNorm2d(1024)

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_dec_1_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=0)
        self.bn_conv_dec_1_1 = nn.BatchNorm2d(512)

        self.conv_dec_1_2 = nn.Conv2d(512, 512, kernel_size=3, padding=0)
        self.bn_conv_dec_1_2 = nn.BatchNorm2d(512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_dec_2_1 = nn.Conv2d(512, 256, kernel_size=3, padding=0)
        self.bn_conv_dec_2_1 = nn.BatchNorm2d(256)

        self.conv_dec_2_2 = nn.Conv2d(256, 256, kernel_size=3, padding=0)
        self.bn_conv_dec_2_2 = nn.BatchNorm2d(256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.bn_conv_dec_3_1 = nn.BatchNorm2d(128)

        
        self.conv_dec_3_1 = nn.Conv2d(256, 128, kernel_size=3, padding=0)
        self.bn_conv_dec_3_1 = nn.BatchNorm2d(128)
        self.conv_dec_3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=0)
        self.bn_conv_dec_3_2 = nn.BatchNorm2d(128)

        self.upconv4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.conv_dec_4_1 = nn.Conv2d(128, 64, kernel_size=3, padding=0)
        self.bn_conv_dec_4_1 = nn.BatchNorm2d(64)
        self.conv_dec_4_2 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.bn_conv_dec_4_2 = nn.BatchNorm2d(64)

        # Also we want to predict segmentation masks for each pixel, so we will use a 1x1 convolutional layer with 2 output channels
        self.conv_dec_5_1 = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.bn_conv_dec_5_1 = nn.BatchNorm2d(1)

        # Instead of 1x1 convolution, we will use dense layers to predict a bounding box
        # first use a global average pooling layer to reduce the size of the output of the last convolutional layer 
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        # Flatten the output of the last convolutional layer
        self.linear_1 = nn.Linear(64,1024)
        # add batch normalization
        self.bn_linear_1 = nn.BatchNorm1d(1024)
        self.linear_2 = nn.Linear(1024,512)
        self.bn_linear_2 = nn.BatchNorm1d(512)
        self.linear_3 = nn.Linear(512,4)





    def forward(self,x):
        x = x.squeeze(1)
        x = self.conv_enc_1_1(x)
        x = self.bn_conv_enc_1_1(x)
        x = self.relu(x)
        x = self.conv_enc_1_2(x)
        x = self.bn_conv_enc_1_2(x)

        x1 = self.relu(x)
        
        x = self.pool(x1)
        x = self.conv_enc_2_1(x)
        x = self.bn_conv_enc_2_1(x)
        x = self.relu(x)
        x = self.conv_enc_2_2(x)
        x = self.bn_conv_enc_2_2(x)

        x2 = self.relu(x)

        x = self.pool(x2)
        x = self.conv_enc_3_1(x)
        x = self.bn_conv_enc_3_1(x)
        x = self.relu(x)
        x = self.conv_enc_3_2(x)
        x = self.bn_conv_enc_3_2(x)

        x3 = self.relu(x)

        x = self.pool(x3)
        x = self.conv_enc_4_1(x)
        x = self.bn_conv_enc_4_1(x)
        x = self.relu(x)
        x = self.conv_enc_4_2(x)
        x = self.bn_conv_enc_4_2(x)

        x4 = self.relu(x)

        x = self.pool(x4)
        x = self.conv_enc_5_1(x)
        x = self.bn_conv_enc_5_1(x)
        x = self.relu(x)
        x = self.conv_enc_5_2(x)
        x = self.bn_conv_enc_5_2(x)
        x = self.relu(x)

        x = self.upconv1(x)
        
        # We use skip connections here, so we concatenate the output of the eighth convolutional layer with the output of the first upconvolutional layer 
        # We should also crop the output of the eighth convolutional layer to match the size of the output of the first upconvolutional layer
        # We will use square images, so we can use the CenterCrop function from torchvision
        cropped_x4 = torchvision.transforms.CenterCrop(x.shape[2])(x4)
        x = torch.cat((cropped_x4, x), dim=1)

        x = self.conv_dec_1_1(x)
        x = self.bn_conv_dec_1_1(x)
        x = self.relu(x)
        x = self.conv_dec_1_2(x)
        x = self.bn_conv_dec_1_2(x)
        x = self.relu(x)

        x = self.upconv2(x)
        
        cropped_x3 = torchvision.transforms.CenterCrop(x.shape[2])(x3)
        x = torch.cat((cropped_x3, x), dim=1)
        x = self.conv_dec_2_1(x)
        x = self.bn_conv_dec_2_1(x)
        x = self.relu(x)
        x = self.conv_dec_2_2(x)
        x = self.bn_conv_dec_2_2(x)
        x = self.relu(x)

        x = self.upconv3(x)
        
        cropped_x2 = torchvision.transforms.CenterCrop(x.shape[2])(x2)
        x = torch.cat((cropped_x2, x), dim=1)
        x = self.conv_dec_3_1(x)
        x = self.bn_conv_dec_3_1(x)
        x = self.relu(x)
        x = self.conv_dec_3_2(x)
        x = self.bn_conv_dec_3_2(x)
        x = self.relu(x)

        x = self.upconv4(x)
        
        cropped_x1 = torchvision.transforms.CenterCrop(x.shape[2])(x1)
        x = torch.cat((cropped_x1, x), dim=1)
        
        x = self.conv_dec_4_1(x)
        x = self.bn_conv_dec_4_1(x)
        x = self.relu(x)
        x = self.conv_dec_4_2(x)
        x = self.bn_conv_dec_4_2(x)
        x = self.relu(x)    
        
        seg = self.conv_dec_5_1(x)
        # Interpolate the output of the last convolutional layer to the size of the input image
        seg = F.interpolate(seg, size=self.out_size)
        # cast to float


        # Instead of 1x1 convolution, we will use dense layers to predict a bounding box
        # first use a global average pooling layer to reduce the size of the output of the last convolutional layer
        x = self.global_avg_pool(x)


        # Flatten the output of the last convolutional layer
        x = x.view(-1,64)

        x = self.linear_1(x)
        x = self.bn_linear_1(x)
        x = self.relu(x)

        x = self.linear_2(x)
        x = self.bn_linear_2(x)

        x = self.relu(x)
        x = self.linear_3(x)

        return seg,x
