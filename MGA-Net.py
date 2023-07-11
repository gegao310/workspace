import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class BN_Conv2d(nn.Module):
    """
    BN_CONV_RELU
    """
    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False) -> object:
        super(BN_Conv2d, self).__init__()
        
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
            
        )
        

    def forward(self, x):
        
        x1  =self.seq(x)

        #return  F.relu(x1)
        return  x1

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class bn(nn.Module):
    def __init__(self,ch_out):
        super(bn,self).__init__()
        self.bn = nn.Sequential(
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.bn(x)
        return x


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class ECALayer(nn.Module):
    def __init__(self, in_dim, k_size=3):
        super(ECALayer, self).__init__()
        self.channel_in = in_dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class multi_block(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super(multi_block,self).__init__()
        #第一个分支 3*3  输出通道数24
        self.branch3x3_1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16,24,kernel_size=3,padding="same")
        self.branch3x3_bn = bn(24)
        #第二个分支 5*5  输出通道数24
        self.branch5x5_1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16,24,kernel_size=3,padding="same")
        self.branch5x5_bn1 = bn(24)
        self.branch5x5_3 = torch.nn.Conv2d(24,24,kernel_size=3,padding="same")
        self.branch5x5_bn2 = bn(24)
        #第三个分支 7*7  输出通道数24
        self.branch7x7_1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)        
        self.branch7x7_2 = torch.nn.Conv2d(16,24,kernel_size=3,padding="same")
        self.branch7x7_bn1 = bn(24)
        self.branch7x7_3 = torch.nn.Conv2d(24,24,kernel_size=3,padding="same")
        self.branch7x7_bn2 = bn(24)
        self.branch7x7_4 = torch.nn.Conv2d(24,24,kernel_size=3,padding="same")
        self.branch7x7_bn3 = bn(24)
        #第四个分支 pool 输出通道数 24
        self.branch_pool = torch.nn.Conv2d(in_channels,24,kernel_size=1)

        self.Conv1x1 = torch.nn.Conv2d(72,out_channels,kernel_size=1)
 
    def forward(self,x):  
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_bn(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_bn1(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)
        branch5x5 = self.branch5x5_bn2(branch5x5)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_bn1(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7) 
        branch7x7 = self.branch7x7_bn2(branch7x7) 
        branch7x7 = self.branch7x7_4(branch7x7)  
        branch7x7 = self.branch7x7_bn3(branch7x7)    

        #branch_pool = F.avg_pool2d(x,kernel_size=3,stride = 1,padding = 1)
        #branch_pool = self.branch_pool(branch_pool)
        output = torch.cat((branch7x7,branch5x5,branch3x3),dim=1)
        #output = [branch7x7,branch5x5,branch3x3]
        output = self.Conv1x1(output)

        #return torch.cat(output,dim=1)
        return output

class GPG_1(nn.Module):
    def __init__(self, in_channels, width=64,norm_layer=nn.BatchNorm2d):
        super(GPG_1, self).__init__()
        

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels[-5], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(5*width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))
        
        self.dilation1 = nn.Sequential(nn.Conv2d(5*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(nn.Conv2d(5*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(nn.Conv2d(5*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(nn.Conv2d(5*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation5 = nn.Sequential(nn.Conv2d(5*width, width, kernel_size=3, padding=16, dilation=16, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        
        self.ECA=ECALayer(width)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]),self.conv3(inputs[-3]),self.conv2(inputs[-4]),self.conv1(inputs[-5])]
        #feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]),self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w))
        feats[-3] = F.interpolate(feats[-3], (h, w))
        feats[-4] = F.interpolate(feats[-4], (h, w))
        feats[-5] = F.interpolate(feats[-5], (h, w))
        
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.ECA(self.dilation1(feat)), self.ECA(self.dilation2(feat)), self.ECA(self.dilation3(feat)), self.ECA(self.dilation4(feat)),self.ECA(self.dilation5(feat))],dim=1)
        feat=self.conv_out(feat)
        return feat

class GPG_2(nn.Module):
    def __init__(self, in_channels, width=512,norm_layer=nn.BatchNorm2d):
        super(GPG_2, self).__init__()       

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        

        self.conv_out = nn.Sequential(
            nn.Conv2d(4*width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))
        
        self.dilation1 = nn.Sequential(nn.Conv2d(4*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(nn.Conv2d(4*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(nn.Conv2d(4*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(nn.Conv2d(4*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        
        self.ECA=ECALayer(width)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]),self.conv3(inputs[-3]),self.conv2(inputs[-4])]
        #feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]),self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w))
        feats[-3] = F.interpolate(feats[-3], (h, w))
        feats[-4] = F.interpolate(feats[-4], (h, w))
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.ECA(self.dilation1(feat)), self.ECA(self.dilation2(feat)), self.ECA(self.dilation3(feat)), self.ECA(self.dilation4(feat))], dim=1)
        feat=self.conv_out(feat)
        return feat

class GPG_3(nn.Module):
    def __init__(self, in_channels, width=512,norm_layer=nn.BatchNorm2d):
        super(GPG_3, self).__init__()      

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(3*width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))
        
        self.dilation1 = nn.Sequential(nn.Conv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(nn.Conv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(nn.Conv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.ECA=ECALayer(width)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w))
        feats[-3] = F.interpolate(feats[-3], (h, w))
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.ECA(self.dilation1(feat)), self.ECA(self.dilation2(feat)), self.ECA(self.dilation3(feat))], dim=1)
        feat=self.conv_out(feat)
        return feat

class GPG_4(nn.Module):
    def __init__(self, in_channels, width=512,norm_layer=nn.BatchNorm2d):
        super(GPG_4, self).__init__()
       
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(2*width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))
        
        self.dilation1 = nn.Sequential(nn.Conv2d(2*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.ECA=ECALayer(width)
        self.dilation2 = nn.Sequential(nn.Conv2d(2*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))     
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w))
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.ECA(self.dilation1(feat)), self.ECA(self.dilation2(feat))], dim=1)
        feat=self.conv_out(feat)
        return feat

class AttU_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=2):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.multi1 = multi_block(in_channels=img_ch,out_channels=64)
        self.ECA1=ECALayer(64)
        self.multi2 = multi_block(in_channels=64,out_channels=128)
        self.ECA2=ECALayer(128)
        self.multi3 = multi_block(in_channels=128,out_channels=256)
        self.ECA3=ECALayer(256)
        self.multi4 = multi_block(in_channels=256,out_channels=512)
        self.ECA4=ECALayer(512)
        self.multi5 = multi_block(in_channels=512,out_channels=1024) 
        self.ECA5=ECALayer(1024)  
    
        self.mce_1=GPG_1([64,128,256, 512, 1024],width=64)
        self.mce_2=GPG_2([128,256, 512, 1024],width=128)
        self.mce_3=GPG_3([256, 512, 1024],width=256)
        self.mce_4=GPG_4([512, 1024],width=512) 

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.multi6 = multi_block(in_channels=1024,out_channels=512) 
        self.ECA6=ECALayer(512)
    
        #self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.multi7 = multi_block(in_channels=512,out_channels=256) 
        self.ECA7=ECALayer(256)
        #self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.multi8 = multi_block(in_channels=256,out_channels=128) 
        self.ECA8=ECALayer(128)
        #self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.multi9 = multi_block(in_channels=128,out_channels=64) 
        self.ECA9=ECALayer(64)
        #self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.multi1(x)
        x1 = self.ECA1(x1)

        x2 = self.Maxpool(x1)
        x2 = self.multi2(x2)
        x2 = self.ECA2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.multi3(x3)
        x3 = self.ECA3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.multi4(x4)
        x4 = self.ECA4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.multi5(x5)
        x5 = self.ECA5(x5)

        # m1=self.mce_1(x1,x2,x3,x4,x5)
        m2=self.mce_2(x2,x3,x4,x5)
        m3=self.mce_3(x3,x4,x5)
        m4=self.mce_4(x4,x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=m4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.multi6(d5)
        d5 = self.ECA6(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=m3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.multi7(d4)
        d4 = self.ECA7(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=m2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.multi8(d3)
        d3 = self.ECA8(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.multi9(d2)
        d2 = self.ECA9(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1,dim=1)
        return d1


