import torch 
import torch.nn as nn
from torch.nn import functional as F
import math

class Model(nn.Module):
            
    def __init__(self, height, width, backbone):
        super(Model, self).__init__()
                
        self.height = height
        self.width = width
        self.n_size = (5, 5, 5, 5, 5)
        
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        
        self.stage1_1 = nn.Sequential(*list(backbone.features.children())[:4])
        
        self.stage1_2 = nn.Sequential(*list(backbone.features.children())[4:9])

        self.stage1_3 = nn.Sequential(*list(backbone.features.children())[9:16])
        
        self.stage1_4 = nn.Sequential(*list(backbone.features.children())[16:23])

        self.stage1_5 = nn.Sequential(*list(backbone.features.children())[23:30])
        

        self.module1 = (self.get_module(n_size = self.n_size[0], n_conv = 3, n_dconv = 0, ch = [64, 64, 64]))
        self.module2 = (self.get_module(n_size = self.n_size[1], n_conv = 3, n_dconv = 1, ch = [64, 64, 64, 64]))
        self.module3 = (self.get_module(n_size = self.n_size[2], n_conv = 3, n_dconv = 2, ch = [64, 64, 64, 64, 64]))
        self.module4 = (self.get_module(n_size = self.n_size[3], n_conv = 3, n_dconv = 3, ch = [64, 64, 64, 64, 64, 64]))
        self.module5 = (self.get_module(n_size = self.n_size[4], n_conv = 3, n_dconv = 4, ch = [64, 64, 64, 64, 64, 64, 64]))
        # self.module6 = self.get_module(n_size = self.n_size[5], n_conv = 1, n_dconv = 2, ch = [2048,64,32])
        
        
        self.ct_conv1 = (nn.Sequential(
                    nn.Conv2d(320, 128, kernel_size=1, stride=1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU()))
        
        self.ct_conv2 = (nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()))
        
        self.ct_conv3 = (nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()))
        
        self.ct_conv4 = (nn.Sequential(
            nn.Conv2d(32, 10, kernel_size=5, stride=1, padding=2)))
        
    def forward(self, img1, img2):
        
        layer1_1 = self.stage1_1(img1)
        layer1_2 = self.stage1_1(img2)
        corr1, depth1 = self.correlation_layer(layer1_1, layer1_2, self.n_size[0], int(self.height), int(self.width))
        
        layer2_1 = self.stage1_2(layer1_1)
        layer2_2 = self.stage1_2(layer1_2)
        corr2, depth2 = self.correlation_layer(layer2_1, layer2_2, self.n_size[1], int(self.height/2), int(self.width/2))
        
        layer3_1 = self.stage1_3(layer2_1)
        layer3_2 = self.stage1_3(layer2_2)
        corr3, depth3 = self.correlation_layer(layer3_1, layer3_2, self.n_size[2], int(self.height/4), int(self.width/4))
        
        layer4_1 = self.stage1_4(layer3_1)
        layer4_2 = self.stage1_4(layer3_2)
        corr4, depth4 = self.correlation_layer(layer4_1, layer4_2, self.n_size[3], int(self.height/8), int(self.width/8))
        
        layer5_1 = self.stage1_5(layer4_1)
        layer5_2 = self.stage1_5(layer4_2)
        corr5, depth5 = self.correlation_layer(layer5_1, layer5_2, self.n_size[4], int(self.height/16), int(self.width/16))
        
        # layer6_1 = self.stage1_6(layer5_1)
        # layer6_2 = self.stage1_6(layer5_2)
        # corr6, depth6 = self.correlation_layer(layer6_1, layer6_2, self.n_size[5], int(self.height/4), int(self.width/4))
        
        l_mod1 = self.module1(corr1)
        l_mod2 = self.module2(corr2)
        l_mod3 = self.module3(corr3)
        l_mod4 = self.module4(corr4)
        l_mod5 = self.module5(corr5)
        # l_mod6 = self.module6(corr6)
        
        ct_layer = torch.cat((l_mod1, l_mod2, l_mod3, l_mod4, l_mod5), dim=1)
        
        out = self.ct_conv1(ct_layer)
        out = self.ct_conv2(out)
        out = self.ct_conv3(out)
        out = self.ct_conv4(out)
        
        return out
    
    def get_depths(self, n_size):
        max_displacement = int(math.ceil(n_size/2.0))
        stride_2 = 2
        assert(stride_2 <= n_size)
        depth = int(math.floor(((2.0 * max_displacement) + 1) / stride_2) ** 2)
        return depth
    
    def get_module(self, n_size, n_conv, n_dconv, ch):
        depth = self.get_depths(n_size)
        assert(n_conv + n_dconv == len(ch))
        ch = [depth, *ch]
        module = []        
        for i in range(len(ch)-1):
            
            if i < n_dconv:
                module.append(nn.ConvTranspose2d(ch[i],ch[i+1], kernel_size=2, stride=2))
                module.append(nn.BatchNorm2d(ch[i+1]))
#                 print("dconv", ch[i], ch[i+1])
            else:
                module.append(nn.Conv2d(ch[i], ch[i+1], kernel_size=5, stride=1, padding=2))
                module.append(nn.BatchNorm2d(ch[i+1]))
                module.append(nn.LeakyReLU())
#                 print("conv", ch[i], ch[i+1])
        return nn.Sequential(*module)
    
    def correlation_layer(self, map1, map2, n_size, h, w):
        HEIGHT = int(h)
        WIDTH = int(w)

        max_displacement = int(math.ceil(n_size/2.0))
        stride_2 = 2
        assert(stride_2 <= n_size)
        depth = int(math.floor(((2.0 * max_displacement) + 1) / stride_2) ** 2)
#         print(depth)
        out = []

        for i in range(-max_displacement+1, max_displacement, stride_2):
            for j in range(-max_displacement+1, max_displacement, stride_2):
                
                padded_a = F.pad(map1, (0,abs(j),0,abs(i)), mode='constant', value=0)
#                 padded_a = nn.ConstantPad2d((0,abs(j),0,abs(i)), 0)(map1)
                
                padded_b = F.pad(map2, (abs(j),0,abs(i),0), mode='constant', value=0)
#                 padded_b = nn.ConstantPad2d((abs(j),0,abs(i),0), 0)(map2)
#                 print(padded_a.shape, padded_b.shape)
                m = padded_a * padded_b
#                 print(m.shape)
                height_start_idx = 0 if i <= 0 else i
                height_end_idx = height_start_idx + HEIGHT
                width_start_idx = 0 if j <= 0 else j
                width_end_idx = width_start_idx + WIDTH
                cut = m[:, :, height_start_idx:height_end_idx, width_start_idx:width_end_idx]
#                 print("c", cut.shape)

                final = torch.sum(cut, 1)
                out.append(final)
        # for o in out:
        #     print(o.shape, n_size)
        corr = torch.stack(out, 1)
        return corr, depth
    
    