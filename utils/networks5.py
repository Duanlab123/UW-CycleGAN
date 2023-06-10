import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import numpy as np

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, middle_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != middle_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels * BottleNeck.expansion, stride=stride, kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(middle_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU()
        )
        self.second = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.first(x)
        out = self.second(out)

        return out


class BetaNet(nn.Module):
    def __init__(self, block, layers, num_classes, input_channels=3,min_beta=0.04, max_beta=0.8):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.min_beta,self.max_beta = min_beta,max_beta
        self.in_channel = nb_filter[0]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[1], layers[0], 1)
        self.conv2_0 = self._make_layer(block, nb_filter[2], layers[1], 1)
        self.conv3_0 = self._make_layer(block, nb_filter[3], layers[2], 1)
        self.conv4_0 = self._make_layer(block, nb_filter[4], layers[3], 1)
        self.conv3_1 = VGGBlock((nb_filter[3] + nb_filter[4]) * block.expansion, nb_filter[3],
                                nb_filter[3] * block.expansion)
        self.conv2_2 = VGGBlock((nb_filter[2] + nb_filter[3]) * block.expansion, nb_filter[2],
                                nb_filter[2] * block.expansion)
        self.conv1_3 = VGGBlock((nb_filter[1] + nb_filter[2]) * block.expansion, nb_filter[1],
                                nb_filter[1] * block.expansion)
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        use_spectral_norm = False
        self.final_conv_beta_1 = conv_block(in_nc=nb_filter[1] + nb_filter[2] + nb_filter[3] + nb_filter[4], out_nc=2 * nb_filter[0], kernel_size=1,
                                            act_type=None, norm_type=None, use_spectral_norm=use_spectral_norm)
        self.final_conv_beta_2 = conv_block(in_nc=2*nb_filter[0], out_nc=3, kernel_size=1, act_type=None, norm_type=None, use_spectral_norm=use_spectral_norm)
    def _make_layer(self, block, middle_channel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, middle_channel, stride))
            self.in_channel = middle_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        layer_1_beta = F.adaptive_avg_pool2d(x1_0, [1, 1])
        layer_2_beta = F.adaptive_avg_pool2d(x2_0, [1, 1])
        layer_3_beta = F.adaptive_avg_pool2d(x3_0, [1, 1])
        layer_4_beta = F.adaptive_avg_pool2d(x4_0, [1, 1])
        beta = self.final_conv_beta_1(torch.cat([layer_1_beta, layer_2_beta, layer_3_beta, layer_4_beta], dim=1))
        beta = self.final_conv_beta_2(beta)
        beta = beta.clamp(self.min_beta,self.max_beta)
        return beta

class Transnet(nn.Module):
    def __init__(self, block, layers, num_classes, input_channels=3):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.in_channel = nb_filter[0]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[1], layers[0], 1)
        self.conv2_0 = self._make_layer(block, nb_filter[2], layers[1], 1)
        self.conv3_0 = self._make_layer(block, nb_filter[3], layers[2], 1)
        self.conv4_0 = self._make_layer(block, nb_filter[4], layers[3], 1)

        self.conv3_1 = VGGBlock((nb_filter[3] + nb_filter[4]) * block.expansion, nb_filter[3],
                                nb_filter[3] * block.expansion)
        self.conv2_2 = VGGBlock((nb_filter[2] + nb_filter[3]) * block.expansion, nb_filter[2],
                                nb_filter[2] * block.expansion)
        self.conv1_3 = VGGBlock((nb_filter[1] + nb_filter[2]) * block.expansion, nb_filter[1],
                                nb_filter[1] * block.expansion)
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, middle_channel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, middle_channel, stride))
            self.in_channel = middle_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        output = output.clamp(0.05,0.99)
        return output


class Lightnet(nn.Module):
    def __init__(self, block, layers, num_classes, input_channels=3):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.in_channel = nb_filter[0]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[1], layers[0], 1)
        self.conv2_0 = self._make_layer(block, nb_filter[2], layers[1], 1)
        self.conv3_0 = self._make_layer(block, nb_filter[3], layers[2], 1)
        self.conv4_0 = self._make_layer(block, nb_filter[4], layers[3], 1)

        self.conv3_1 = VGGBlock((nb_filter[3] + nb_filter[4]) * block.expansion, nb_filter[3],
                                nb_filter[3] * block.expansion)
        self.conv2_2 = VGGBlock((nb_filter[2] + nb_filter[3]) * block.expansion, nb_filter[2],
                                nb_filter[2] * block.expansion)
        self.conv1_3 = VGGBlock((nb_filter[1] + nb_filter[2]) * block.expansion, nb_filter[1],
                                nb_filter[1] * block.expansion)
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, middle_channel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, middle_channel, stride))
            self.in_channel = middle_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        output = output.clamp(0.2,1)
        return output

class Depthnet(nn.Module):
    def __init__(self, block, layers, num_classes, input_channels=3):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.in_channel = nb_filter[0]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[1], layers[0], 1)
        self.conv2_0 = self._make_layer(block, nb_filter[2], layers[1], 1)
        self.conv3_0 = self._make_layer(block, nb_filter[3], layers[2], 1)
        self.conv4_0 = self._make_layer(block, nb_filter[4], layers[3], 1)

        self.conv3_1 = VGGBlock((nb_filter[3] + nb_filter[4]) * block.expansion, nb_filter[3],
                                nb_filter[3] * block.expansion)
        self.conv2_2 = VGGBlock((nb_filter[2] + nb_filter[3]) * block.expansion, nb_filter[2],
                                nb_filter[2] * block.expansion)
        self.conv1_3 = VGGBlock((nb_filter[1] + nb_filter[2]) * block.expansion, nb_filter[1],
                                nb_filter[1] * block.expansion)
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, middle_channel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, middle_channel, stride))
            self.in_channel = middle_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        output = output.clamp(0.3,15)
        return output
class Cleannet(nn.Module):
    def __init__(self, block, layers, num_classes, input_channels=3):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.in_channel = nb_filter[0]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[1], layers[0], 1)
        self.conv2_0 = self._make_layer(block, nb_filter[2], layers[1], 1)
        self.conv3_0 = self._make_layer(block, nb_filter[3], layers[2], 1)
        self.conv4_0 = self._make_layer(block, nb_filter[4], layers[3], 1)

        self.conv3_1 = VGGBlock((nb_filter[3] + nb_filter[4]) * block.expansion, nb_filter[3],
                                nb_filter[3] * block.expansion)
        self.conv2_2 = VGGBlock((nb_filter[2] + nb_filter[3]) * block.expansion, nb_filter[2],
                                nb_filter[2] * block.expansion)
        self.conv1_3 = VGGBlock((nb_filter[1] + nb_filter[2]) * block.expansion, nb_filter[1],
                                nb_filter[1] * block.expansion)
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, middle_channel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, middle_channel, stride))
            self.in_channel = middle_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        output = output.clamp(0.05,1)
        return output


class HazeProduceNet(nn.Module):
    def __init__(self,basenorm=48):
        super().__init__()
        act_type = 'leakyrelu'
        norm_type = 'batch'
        mode = 'CNA'
        use_spectral_norm = False
    def forward(self, x, d,beta,A):
        # A = x.max(dim=3)[0].max(dim=2, keepdim=True)[0].unsqueeze(3)
        #t = torch.exp(-d*beta)
        t = torch.exp(torch.mul(-d, beta))
        x = torch.mul(t, x) + torch.mul(A, 1 - t)
  
        return x
    def forward_random_parameters(self, image, depth,water_type=1,water_depth=0,water_illum=[1,0]):  # x:NCHW, ex:e^(-d(x)), N,1,H,W, beta:N,1,1,1
        eta_r = np.array([0.30420412, 0.30474395, 0.35592191, 0.32493874, 0.55091001,0.42493874, 0.55874165, 0.13039252 , 0.10760831, 0.15963731])
        eta_g = np.array([0.11727661, 0.05999663, 0.11227639, 0.15305673, 0.14385827, 0.12305673, 0.0518615, 0.18667714, 0.1567016, 0.205724217])
        eta_b = np.array([0.1488851, 0.30099538, 0.38412464, 0.25060999, 0.01387215, 0.055799195, 0.0591001, 0.5539252 , 0.60103   , 0.733602  ])
        depth = depth.detach().cpu().numpy()
        image = image.detach().cpu().numpy()
        water_type = int(np.random.random() * len(eta_r))
        water_depth = np.random.uniform(0.4, 10, 1)
        water_illum = np.random.uniform(0, 1, 2)
        water_illum = water_illum / sum(water_illum)
        depth = depth + water_depth
        #---------------------------------------------------------------------------------------------------------------
        eta_rI, eta_gI, eta_bI = torch.tensor([[[eta_r[water_type]]]]), torch.tensor([[[eta_g[water_type]]]]), torch.tensor([[[eta_b[water_type]]]])
        eta1 = torch.stack([eta_rI, eta_gI, eta_bI], axis=3)
        water_type2 = 8 if water_type + 1 == 10 else water_type + 1
        eta_rI, eta_gI, eta_bI = torch.tensor([[[eta_r[water_type2]]]]), torch.tensor( [[[eta_g[water_type2]]]]), torch.tensor([[[eta_b[water_type2]]]])
        eta2 = torch.stack([eta_rI, eta_gI, eta_bI], axis=3)
        weight = torch.from_numpy(np.random.uniform(0, 1, 1))
        eta = weight * eta1 + (1 - weight) * eta2
        eta = eta.detach().numpy()
        eta = eta.transpose((0, 3, 1, 2))
        t1=np.ones((image.shape[0],1,1,1))
        eta = t1* eta
        #-------------------------------------------------------
        t = np.exp(np.multiply(-1, np.multiply(depth, eta)))
        Z_b = np.random.uniform(0, 2)
        L_t1 =np.exp(  np.multiply(-water_depth, eta) )
        L_t2 = np.exp(np.multiply(-1, np.multiply(depth + Z_b-water_depth, eta)))
        L_t = water_illum[0] * L_t1 + water_illum[1] * L_t2  # L_t3
        direct = np.multiply(image, t)
        #---------------------------------------------------------------
        #t_haze = np.exp(np.multiply(-1, np.multiply(depth, eta)))
        A = np.random.uniform(0.8, 1.1, 1) * L_t
        image_haze = np.multiply(A, 1 - t)
        B = A
        I = direct + image_haze
        I = torch.from_numpy(I).cuda().float()
        B = torch.from_numpy(B).cuda().float()
        t = torch.from_numpy(t).cuda().float()
        eta = torch.from_numpy(eta).cuda().float()
        #res1 = self(x, d, beta)
        return I, eta, B,t
def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):

    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA', use_spectral_norm=False):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """
    assert mode in ['CNA', 'NAC', 'CNAC', 'CAN'], 'Wrong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = spectral_norm(nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups), mode=use_spectral_norm)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)

    elif mode == 'CAN':
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, a, n)


def deconv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, padding=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wrong conv mode [{:s}]'.format(mode)

    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.ConvTranspose2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)
def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

if __name__ == '__main__':
    UResnet34 = BetaNet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=3)
    print(UResnet34)
    x=torch.rand((2,3,256,256))
    t1= UResnet34.forward(x)
    UResnet50 = BetaNet(block=BottleNeck, layers=[3, 4, 6, 3], num_classes=2)

    UReset101 = BetaNet(block=BottleNeck, layers=[3, 4, 23, 3], num_classes=2)

    UReset152 = BetaNet(block=BottleNeck, layers=[3, 8, 36, 3], num_classes=2)