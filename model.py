import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, eps = 1e-5, momentum = 0.1 ),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, eps = 1e-5, momentum = 0.1))

    def forward(self, x):
        return x + self.main(x)
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y.expand_as(x)
    

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
    
class Attention(nn.Sequential):
    """ Attention layer """
    def __init__(self, num_input_feature):
        super(Attention, self).__init__()
        self.add_module("conv", nn.Conv2d(num_input_feature, 1, kernel_size=7, 
                                          stride=1, padding=3, bias=False))
        self.add_module("Sigmoid", nn.Sigmoid())
        
    def forward(self, x):
        attention_map = super(Attention, self).forward(x)
        attentioned_feature_map = x * attention_map
        return attentioned_feature_map
    
    
class Line(nn.Module):
    '''Line nework.''' 
    def __init__(self, conv_dim = 64, repeat_num = 9, down_plies = 2, curr_dim = None):
        super(Line, self).__init__()
        '''Line'''
        #########################################   line   ########################################
        layers = []       #channel 1 --> 64   #128*128 --> 128*128
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.BatchNorm2d(conv_dim, eps = 1e-5, momentum = 0.1 ),)
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        # Dimensions / 4
        curr_dim = conv_dim
        for i in range(down_plies):     #curr_dim = 64 --> 128 --> 256     #128*128 --> 64*64 --> 32*32
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(curr_dim*2, eps = 1e-5, momentum = 0.1 ),)
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        # Dimensions are the same 
        for i in range(repeat_num):       #curr_dim = 256 --> ~~ --> 256   #32*32 --> ~~ --> 32*32
            # (ResNet)
            # layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
            # (SEBasicBlock)
            layers.append(SEBasicBlock(inplanes = curr_dim, planes = curr_dim))
    
        self.line = nn.Sequential(*layers)
        
    def forward(self, img):
        Feature = self.line(img)
        return Feature

class Generator(nn.Module):
    '''Generator network.'''
    def __init__(self, conv_dim = 64, repeat_num = 9, model = 'Gen', down_plies = 2, up_plies = 2):  # conv_dim = 64
        super(Generator, self).__init__()
        '''Generator'''
        #########################################   encode layer   ########################################
        layers1 = []
        layers1.append(Line(conv_dim = 64, repeat_num = 9, down_plies = 2, curr_dim = None))
        self.line1 = nn.Sequential(*layers1)
        curr_dim = conv_dim * 2 * down_plies   #curr_dim = 256

        #########################################   decode layer   ########################################

        layers3 = []
        # Up-sampling layers.
        # Dimensions * 4
        curr_dim = curr_dim * 2    #  curr_dim = 512
        for i in range(up_plies):    #curr_dim =512 --> 256 --> 128   #32*32 --> 64*64 --> 128*128
            layers3.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers3.append(nn.BatchNorm2d(curr_dim//2, eps = 1e-5, momentum = 0.1 ))
            layers3.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
            
                                           #curr_dim =128 --> 1    # 128*128 --> 128*128
        layers3.append(nn.Conv2d(curr_dim, 1, kernel_size = 7, stride=1, padding = 3, bias = False ))
        layers3.append(nn.Sigmoid())
        self.decode = nn.Sequential(*layers3)
                                   
    def forward(self, A, B):
        Ar, Ag, Ab = torch.chunk(A, 3, dim = 1)
        Br, Bg, Bb = torch.chunk(B, 3, dim = 1)
        Feature_Ar = self.line1(Ar)
        Feature_Ag = self.line1(Ag)
        Feature_Ab = self.line1(Ab)
        Feature_Br = self.line1(Br)
        Feature_Bg = self.line1(Bg)
        Feature_Bb = self.line1(Bb)
        Feature_A = (Feature_Ar + Feature_Ag + Feature_Ab) / 3 
        Feature_B = (Feature_Br + Feature_Bg + Feature_Bb) / 3 
        concatenation = torch.cat([Feature_A, Feature_B], dim = 1)
        decision_map = self.decode(concatenation)
        return decision_map
        
        
class Discriminator(nn.Module):
    """Discriminator network."""
    def __init__(self, image_size=128, conv_dim=64, repeat_num=3): #c_dim=5, # conv_dim = 64
        super(Discriminator, self).__init__()
        layers = []   # 7 --> 64     # 128 * 128 --> 64 * 64
        layers.append(nn.Conv2d(7, conv_dim, kernel_size=4, stride=2, padding=2))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):  # 64 --> 128 --> 256 --> 512   # 64*64 --> 32*32 --> 16*16 --> 8*8 
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(curr_dim*2, eps = 1e-5, momentum = 0.1 ))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        # Dimensions are the same 
        for i in range(repeat_num):    # 512 --> 512 --> 512 --> 512   # 8*8 --> 4*4 --> 2*2 -->1*1
            layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=2, bias=False))
            layers.append(nn.BatchNorm2d( curr_dim, eps = 1e-5, momentum = 0.1 ))
            layers.append(nn.ReLU(inplace=True))
            
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=2))
        layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)
        
    def forward(self, A, B, focus_map):
        Ar, Ag, Ab = torch.chunk(A, 3, dim = 1)
        Br, Bg, Bb = torch.chunk(B, 3, dim = 1)
        imgA = torch.cat([Ar,Ag,Ab], dim = 1)
        imgB = torch.cat([Br,Bg,Bb], dim = 1)
        Img = torch.cat([imgA,imgB, focus_map], dim = 1)
        decision = self.main(Img)

        return decision
