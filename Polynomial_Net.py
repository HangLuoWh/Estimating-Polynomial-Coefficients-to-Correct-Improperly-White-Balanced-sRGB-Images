import torch.nn as nn
import torch

# channel attention module 
class Attention(nn.Module):
    """
    parameter:
        reduction: decay factor to control the number of neurons in the hidden layer
        in_chan: the number of input channel
    """
    def __init__(self, reduction, in_chan):
        super(Attention, self).__init__()
        self.reduction = reduction
        self.in_chan = in_chan
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_net = nn.Sequential(
            nn.Linear(self.in_chan, self.in_chan//self.reduction, bias = True),
            nn.ReLU(inplace = True),
            nn.Linear(self.in_chan//self.reduction, self.in_chan, bias = True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = torch.squeeze(self.avg_pool(x), 2)
        y = torch.squeeze(y, 2)
        attention = self.fc_net(y)
        attention = torch.unsqueeze(attention, dim = 2)
        attention = torch.unsqueeze(attention, dim = 3)
        return x*attention
        
# neuron attention module 
class NeuronAttention(nn.Module):
    """
    parameter:
        reduction: decay factor to control the number of neurons in the hidden layer
        num: the number of neurons in the input layer
    """
    def __init__(self, reduction, num):
        super(NeuronAttention, self).__init__()
        self.reduction = reduction
        self.num = num
        self.main = nn.Sequential(
            nn.Linear(self.num, self.num//self.reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.num//self.reduction, self.num, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        weights = self.main(x)
        return weights*x


class PolyNet(nn.Module):
    def __init__(self, hyper):
        super(PolyNet, self).__init__()
        self.hyper = hyper

        # convolution layers
        convs = []
        for i in range(5):
            if i == 0:
                convs.append(nn.Conv2d(self.hyper['img_chan'], self.hyper['nfc']*(i+1), kernel_size=2, stride=2, padding=0))
            else:
                convs.append(nn.Conv2d(self.hyper['nfc']*i, self.hyper['nfc'] * (i + 1), kernel_size=2, stride=2, padding=0))
            convs.append(nn.PReLU(num_parameters = self.hyper['nfc'] * (i + 1), init = 0.25))
            # add channel attention module or not
            if self.hyper['channel_attention']:
                convs.append(Attention(self.hyper['channel_redu'], self.hyper['nfc'] * (i+1)))
        self.convs = nn.Sequential(*convs)

        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))

        # the number of channels in the feature map
        chan_num = self.hyper['nfc']*5*4

        # add neuron attention module or not
        self.neu_att = NeuronAttention(self.hyper['neuron_redu'], chan_num) if self.hyper['neuron_attention'] else None

        # fully connected layers
        fcs = []
        for i in range(self.hyper['linear_num']):
            if i == 0:
                fcs.append(nn.Linear(chan_num, 1000))
            else:
                fcs.append(nn.Linear(1000, 1000))
            fcs.append(nn.PReLU(num_parameters=1000, init=0.25))
            fcs.append(nn.Dropout(0.5))
        fcs.append(nn.Linear(1000, 3*self.hyper['term']))
        self.fc = nn.Sequential(*fcs)

    def forward(self, img):
        b,c,h,w = img.size()

        x = img
        x = x - 0.5

        con_out = self.avg_pool(self.convs(x)) # 卷积层的输出
        con_out = con_out.view(b, -1)
        
        if self.hyper['neuron_attention']:
            feature = self.neu_att(con_out) # 神经元注意力模型
        else:
            feature = con_out

        para = self.fc(feature)
        para = para.view(para.size(0), -1, 3)
        
        # polynomial extension of root-polynomial extension
        if self.hyper['root']:
            img_ext = self.root_extend(img.detach())
        else:
            img_ext = self.extend(img.detach())

        # image transformation
        img_coarse = torch.einsum('bcij,bcf->bfij', img_ext, para)

        # computer residual or not
        if self.hyper['residual']:
            img_coarse = torch.clamp(img_coarse + x, min = 0, max = 1)
        else:
            img_coarse = torch.clamp(img_coarse, min = 0, max = 1)
        return img_coarse, para
    
    # polynomial extension
    def extend(self, img):
        if self.hyper['term'] == 3: # 3 terms
            return img
        if self.hyper['term'] == 4: # using log terms
            shift = 0.000000000001
            img_r = img[:, 0, :, :]
            img_g = img[:, 1, :, :]
            img_b = img[:, 2, :, :]
            img_lr = torch.log2(img_r + shift)
            img_lg = torch.log2(img_g + shift)
            img_lb = torch.log2(img_b + shift)

            img_lr = torch.unsqueeze(img_lr, 1)
            img_lg = torch.unsqueeze(img_lg, 1)
            img_lb = torch.unsqueeze(img_lb, 1)

            return torch.cat([img, img_lr, img_lg, img_lb], dim = 1)
        elif self.hyper['term'] == 5: #5 terms
            img_r = img[:, 0, :, :]
            img_g = img[:, 1, :, :]
            img_b = img[:, 2, :, :]
            img_rgb = img_r*img_g*img_b
            img_1 = torch.ones_like(img_r)

            img_rgb = torch.unsqueeze(img_rgb, 1)
            img_1 = torch.unsqueeze(img_1, 1)

            return torch.cat([img, img_rgb, img_1], dim = 1)
        elif self.hyper['term'] == 6: #6 terms
            img_r = img[:, 0, :, :]
            img_g = img[:, 1, :, :]
            img_b = img[:, 2, :, :]
            img_rg = img_r * img_g
            img_rb = img_r * img_b
            img_gb = img_g * img_b

            img_rg = torch.unsqueeze(img_rg, 1)
            img_rb = torch.unsqueeze(img_rb, 1)
            img_gb = torch.unsqueeze(img_gb, 1)

            return torch.cat([img, img_rg, img_rb, img_gb], dim = 1)
        elif self.hyper['term'] == 7: #7 terms
            img_r = img[:, 0, :, :]
            img_g = img[:, 1, :, :]
            img_b = img[:, 2, :, :]
            img_rg = img_r * img_g
            img_rb = img_r * img_b
            img_gb = img_g * img_b
            img_1 = torch.ones_like(img_r)

            img_rg = torch.unsqueeze(img_rg, 1)
            img_rb = torch.unsqueeze(img_rb, 1)
            img_gb = torch.unsqueeze(img_gb, 1)
            img_1 = torch.unsqueeze(img_1, 1)

            return torch.cat([img, img_rg, img_rb, img_gb, img_1], dim = 1)
        elif self.hyper['term'] == 9: #9 terms
            img_r = img[:, 0, :, :]
            img_g = img[:, 1, :, :]
            img_b = img[:, 2, :, :]
            img_rg = img_r * img_g
            img_rb = img_r * img_b
            img_gb = img_g * img_b
            img_r2 = torch.pow(img_r, 2)
            img_g2 = torch.pow(img_g, 2)
            img_b2 = torch.pow(img_b, 2)

            img_rg = torch.unsqueeze(img_rg, 1)
            img_rb = torch.unsqueeze(img_rb, 1)
            img_gb = torch.unsqueeze(img_gb, 1)
            img_r2 = torch.unsqueeze(img_r2, 1)
            img_g2 = torch.unsqueeze(img_g2, 1)
            img_b2 = torch.unsqueeze(img_b2, 1)

            return torch.cat([img, img_rg, img_rb, img_gb, img_r2, img_g2, img_b2], dim = 1) 
        elif self.hyper['term'] == 11: # 11 terms
            img_r = img[:, 0, :, :]
            img_g = img[:, 1, :, :]
            img_b = img[:, 2, :, :]
            img_rg = img_r * img_g
            img_rb = img_r * img_b
            img_gb = img_g * img_b
            img_r2 = torch.pow(img_r, 2)
            img_g2 = torch.pow(img_g, 2)
            img_b2 = torch.pow(img_b, 2)
            img_rgb = img_r * img_g * img_b
            img_1 = torch.ones_like(img_r)

            img_rg = torch.unsqueeze(img_rg, 1)
            img_rb = torch.unsqueeze(img_rb, 1)
            img_gb = torch.unsqueeze(img_gb, 1)
            img_r2 = torch.unsqueeze(img_r2, 1)
            img_g2 = torch.unsqueeze(img_g2, 1)
            img_b2 = torch.unsqueeze(img_b2, 1)
            img_rgb = torch.unsqueeze(img_rgb, 1)
            img_1 = torch.unsqueeze(img_1, 1)

            return torch.cat([img, img_rg, img_rb, img_gb, img_r2, img_g2, img_b2, img_rgb, img_1], dim = 1)
        elif self.hyper['term'] == 19: # 19 terms
            img_r = img[:, 0, :, :]
            img_g = img[:, 1, :, :]
            img_b = img[:, 2, :, :]
            img_rg = img_r * img_g
            img_rb = img_r * img_b
            img_gb = img_g * img_b
            img_r2 = torch.pow(img_r, 2)
            img_g2 = torch.pow(img_g, 2)
            img_b2 = torch.pow(img_b, 2)
            img_r3 = torch.pow(img_r, 3)
            img_g3 = torch.pow(img_g, 3)
            img_b3 = torch.pow(img_b, 3)
            img_rg2 = img_r*img_g**2
            img_gb2 = img_g*img_b**2
            img_rb2 = img_r*img_b**2
            img_gr2 = img_g*img_r**2
            img_bg2 = img_b*img_g**2
            img_br2 = img_b*img_r**2
            img_rgb = img_r * img_g * img_b

            img_rg = torch.unsqueeze(img_rg, 1)
            img_rb = torch.unsqueeze(img_rb, 1)
            img_gb = torch.unsqueeze(img_gb, 1)
            img_r2 = torch.unsqueeze(img_r2, 1)
            img_g2 = torch.unsqueeze(img_g2, 1)
            img_b2 = torch.unsqueeze(img_b2, 1)
            img_r3 = torch.unsqueeze(img_r3, 1)
            img_g3 = torch.unsqueeze(img_g3, 1)
            img_b3 = torch.unsqueeze(img_b3, 1)
            img_rg2 = torch.unsqueeze(img_rg2, 1)
            img_gb2 = torch.unsqueeze(img_gb2, 1)
            img_rb2 = torch.unsqueeze(img_rb2, 1)
            img_gr2 = torch.unsqueeze(img_gr2, 1)
            img_bg2 = torch.unsqueeze(img_bg2, 1)
            img_br2 = torch.unsqueeze(img_br2, 1)
            img_rgb = torch.unsqueeze(img_rgb, 1)

            return torch.cat([img, img_r2, img_g2, img_b2, img_rg, img_gb, img_rb, img_r3, img_g3, img_b3, \
                               img_rg2, img_gb2, img_rb2, img_gr2, img_bg2, img_br2, img_rgb], dim = 1)
        elif self.hyper['term'] == 34: # 34 terms
            img_r = img[:, 0, :, :]
            img_g = img[:, 1, :, :]
            img_b = img[:, 2, :, :]
            img_rg = img_r * img_g
            img_rb = img_r * img_b
            img_gb = img_g * img_b
            img_r2 = torch.pow(img_r, 2)
            img_g2 = torch.pow(img_g, 2)
            img_b2 = torch.pow(img_b, 2)
            img_r3 = torch.pow(img_r, 3)
            img_g3 = torch.pow(img_g, 3)
            img_b3 = torch.pow(img_b, 3)
            img_rg2 = img_r*img_g**2
            img_gb2 = img_g*img_b**2
            img_rb2 = img_r*img_b**2
            img_gr2 = img_g*img_r**2
            img_bg2 = img_b*img_g**2
            img_br2 = img_b*img_r**2
            img_rgb = img_r * img_g * img_b

            img_r4 = img_r**4
            img_g4 = img_g**4
            img_b4 = img_b**4
            img_r3g = img_g*img_r**3
            img_r3b = img_b*img_r**3
            img_g3r = img_r*img_g**3
            img_g3b = img_b*img_g**3
            img_b3r = img_r*img_b**3
            img_b3g = img_g*img_b**3
            img_r2g2 = img_r**2*img_g**2
            img_g2b2 = img_g**2*img_b**2
            img_r2b2 = img_r**2*img_b**2
            img_r2gb = img_r**2*img_g*img_b
            img_g2rb = img_g**2*img_r*img_b
            img_b2rg = img_b**2*img_r*img_b

            img_rg = torch.unsqueeze(img_rg, 1)
            img_rb = torch.unsqueeze(img_rb, 1)
            img_gb = torch.unsqueeze(img_gb, 1)
            img_r2 = torch.unsqueeze(img_r2, 1)
            img_g2 = torch.unsqueeze(img_g2, 1)
            img_b2 = torch.unsqueeze(img_b2, 1)
            img_r3 = torch.unsqueeze(img_r3, 1)
            img_g3 = torch.unsqueeze(img_g3, 1)
            img_b3 = torch.unsqueeze(img_b3, 1)
            img_rg2 = torch.unsqueeze(img_rg2, 1)
            img_gb2 = torch.unsqueeze(img_gb2, 1)
            img_rb2 = torch.unsqueeze(img_rb2, 1)
            img_gr2 = torch.unsqueeze(img_gr2, 1)
            img_bg2 = torch.unsqueeze(img_bg2, 1)
            img_br2 = torch.unsqueeze(img_br2, 1)
            img_rgb = torch.unsqueeze(img_rgb, 1)

            img_r4 = torch.unsqueeze(img_r4, 1)
            img_g4 = torch.unsqueeze(img_g4, 1)
            img_b4 = torch.unsqueeze(img_b4, 1)
            img_r3g = torch.unsqueeze(img_r3g, 1)
            img_r3b = torch.unsqueeze(img_r3b, 1)
            img_g3r = torch.unsqueeze(img_g3r, 1)
            img_g3b = torch.unsqueeze(img_g3b, 1)
            img_b3r = torch.unsqueeze(img_b3r, 1)
            img_b3g = torch.unsqueeze(img_b3g, 1)
            img_r2g2 = torch.unsqueeze(img_r2g2, 1)
            img_g2b2 = torch.unsqueeze(img_g2b2, 1)
            img_r2b2 = torch.unsqueeze(img_r2b2, 1)
            img_r2gb = torch.unsqueeze(img_r2gb, 1)
            img_g2rb = torch.unsqueeze(img_g2rb, 1)
            img_b2rg = torch.unsqueeze(img_b2rg, 1)

            return torch.cat([img, img_r2, img_g2, img_b2, img_rg, img_gb, img_rb, img_r3, img_g3, img_b3, \
                               img_rg2, img_gb2, img_rb2, img_gr2, img_bg2, img_br2, img_rgb,\
                                img_r4, img_g4, img_b4, img_r3g, img_r3b, img_g3r, img_g3b, img_b3r, img_b3g,\
                                img_r2g2, img_g2b2, img_r2b2, img_r2gb, img_g2rb, img_b2rg], dim = 1)
        else:
            raise NotImplementedError(f'term {self.hyper["term"]} is not implemented in PolyNet')
    
    # root-polynomial extension
    def root_extend(self, img):
        if self.hyper['term'] == 6: # 6 terms
            img_r = img[:, 0, :, :]
            img_g = img[:, 1, :, :]
            img_b = img[:, 2, :, :]
            img_rg = torch.pow(img_r * img_g, 1/2)
            img_rb = torch.pow(img_r * img_b, 1/2)
            img_gb = torch.pow(img_g * img_b, 1/2)

            img_rg = torch.unsqueeze(img_rg, 1)
            img_rb = torch.unsqueeze(img_rb, 1)
            img_gb = torch.unsqueeze(img_gb, 1)

            return torch.cat([img, img_rg, img_rb, img_gb], dim = 1)

        elif self.hyper['term'] == 13: #13 terms
            img_r = img[:, 0, :, :]
            img_g = img[:, 1, :, :]
            img_b = img[:, 2, :, :]
            img_rg = torch.pow(img_r * img_g, 1/2)
            img_rb = torch.pow(img_r * img_b, 1/2)
            img_gb = torch.pow(img_g * img_b, 1/2)
            img_rg2 = torch.pow(img_r*img_g**2, 1/3)
            img_gb2 = torch.pow(img_g*img_b**2, 1/3)
            img_rb2 = torch.pow(img_r*img_b**2, 1/3)
            img_gr2 = torch.pow(img_g*img_r**2, 1/3)
            img_bg2 = torch.pow(img_b*img_g**2, 1/3)
            img_br2 = torch.pow(img_b*img_r**2, 1/3)
            img_rgb = torch.pow(img_r*img_g*img_b, 1/3)
            
            img_rg = torch.unsqueeze(img_rg, 1)
            img_rb = torch.unsqueeze(img_rb, 1)
            img_gb = torch.unsqueeze(img_gb, 1)
            img_rg2 = torch.unsqueeze(img_rg2, 1)
            img_gb2 = torch.unsqueeze(img_gb2, 1)
            img_rb2 = torch.unsqueeze(img_rb2, 1)
            img_gr2 = torch.unsqueeze(img_gr2, 1)
            img_bg2 = torch.unsqueeze(img_bg2, 1)
            img_br2 = torch.unsqueeze(img_br2, 1)
            img_rgb = torch.unsqueeze(img_rgb, 1)

            return torch.cat([img, img_rg, img_rb, img_gb, img_rg2, img_gb2, \
                img_rb2, img_gr2, img_bg2, img_br2, img_rgb], dim = 1)

        elif self.hyper['term'] == 22: #22 terms
            img_r = img[:, 0, :, :]
            img_g = img[:, 1, :, :]
            img_b = img[:, 2, :, :]
            img_rg = torch.pow(img_r * img_g, 1/2)
            img_rb = torch.pow(img_r * img_b, 1/2)
            img_gb = torch.pow(img_g * img_b, 1/2)
            img_rg2 = torch.pow(img_r*img_g**2, 1/3)
            img_gb2 = torch.pow(img_g*img_b**2, 1/3)
            img_rb2 = torch.pow(img_r*img_b**2, 1/3)
            img_gr2 = torch.pow(img_g*img_r**2, 1/3)
            img_bg2 = torch.pow(img_b*img_g**2, 1/3)
            img_br2 = torch.pow(img_b*img_r**2, 1/3)
            img_rgb = torch.pow(img_r*img_g*img_b, 1/3)
            img_r3g = torch.pow(img_r**3*img_g, 1/4)
            img_r3b = torch.pow(img_r**3*img_b, 1/4)
            img_g3r = torch.pow(img_g**3*img_r, 1/4)
            img_g3b = torch.pow(img_g**3*img_b, 1/4)
            img_b3r = torch.pow(img_b**3*img_r, 1/4)
            img_b3g = torch.pow(img_b**3*img_g, 1/4)
            img_r2gb = torch.pow(img_r**2*img_g*img_b, 1/4)
            img_g2rb = torch.pow(img_g**2*img_r*img_b, 1/4)
            img_b2rg = torch.pow(img_b**2*img_r*img_g, 1/4)
            
            img_rg = torch.unsqueeze(img_rg, 1)
            img_rb = torch.unsqueeze(img_rb, 1)
            img_gb = torch.unsqueeze(img_gb, 1)
            img_rg2 = torch.unsqueeze(img_rg2, 1)
            img_gb2 = torch.unsqueeze(img_gb2, 1)
            img_rb2 = torch.unsqueeze(img_rb2, 1)
            img_gr2 = torch.unsqueeze(img_gr2, 1)
            img_bg2 = torch.unsqueeze(img_bg2, 1)
            img_br2 = torch.unsqueeze(img_br2, 1)
            img_rgb = torch.unsqueeze(img_rgb, 1)

            img_r3g = torch.unsqueeze(img_r3g, 1)
            img_r3b = torch.unsqueeze(img_r3b, 1)
            img_g3r = torch.unsqueeze(img_g3r, 1)
            img_g3b = torch.unsqueeze(img_g3b, 1)
            img_b3r = torch.unsqueeze(img_b3r, 1)
            img_b3g = torch.unsqueeze(img_b3g, 1)
            img_r2gb = torch.unsqueeze(img_r2gb, 1)
            img_g2rb = torch.unsqueeze(img_g2rb, 1)
            img_b2rg = torch.unsqueeze(img_b2rg, 1)

            return torch.cat([img, img_rg, img_rb, img_gb, img_rg2, img_gb2, \
                img_rb2, img_gr2, img_bg2, img_br2, img_rgb, img_r3g, img_r3b, \
                img_g3r, img_g3b, img_b3r, img_b3g, img_r2gb, img_g2rb, img_b2rg], dim = 1)