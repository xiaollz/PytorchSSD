import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .VGG16 import get_vgg16_fms

from .cbam import *

import ipdb

##
# Date:11/21
# reduce the channel of FM3
#
class BasicConv(nn.Module):
    # add bn
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=True, up_size=0):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = F.interpolate(x, size=(self.up_size, self.up_size), mode='bilinear', align_corners=True)
        return x

def add_extras(cfg, i, batch_norm=False, size=512):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return layers

# 38x38
def feature_transform_module_1(size):
    if size ==  300:
        up_size = 38
    elif size == 512:
        up_size = 64

    layers = []
    # conv4_3
    layers += [BasicConv(512, 256, kernel_size=1, padding=0)]
    # fc_7
    layers += [BasicConv(1024, 256, kernel_size=1, padding=0, up_size=up_size)]
    layers += [BasicConv(512, 256, kernel_size=1, padding=0, up_size=up_size)] #256->512
    return layers

# 19x19
def feature_transform_module_2(size):
    if size ==  300:
        up_size = 19
    elif size == 512:
        up_size = 32

    layers = []
    # fc_7
    layers += [BasicConv(1024, 256, kernel_size=1, padding=0)]
    # conv6_2
    layers += [BasicConv(512, 256, kernel_size=1, padding=0, up_size=up_size)]
    # conv7_2
    layers += [BasicConv(256, 256, kernel_size=1, padding=0, up_size=up_size)]
    return layers

# 10x10

##
# Date: 11/20
# 256->128
#
def feature_transform_module_3(size):
    if size ==  300:
        up_size = 10
    elif size == 512:
        up_size = 16

    layers = []
    # conv6_2
    layers += [BasicConv(512, 128, kernel_size=1, padding=0)]
    # conv7_2
    layers += [BasicConv(256, 128, kernel_size=1, padding=0, up_size=up_size)]
    # conv8_2
    layers += [BasicConv(256, 128, kernel_size=1, padding=0, up_size=up_size)]
    return layers

def pyramid_feature_extractor_1(size):
    if size == 300:
        layers = [BasicConv(256 * 3, 512, kernel_size=3, stride=1, padding=1),
                  BasicConv(512, 512, kernel_size=3, stride=2, padding=1), \
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=1, padding=0),
                  BasicConv(256, 256, kernel_size=3, stride=1, padding=0)]
    elif size == 512:
        layers = [BasicConv(256 * 3, 512, kernel_size=3, stride=1, padding=1),
                  BasicConv(512, 512, kernel_size=3, stride=2, padding=1), \
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=4, padding=1, stride=1)]
    return layers

def pyramid_feature_extractor_2(size):
    if size == 300:
        layers = [BasicConv(256 * 3, 512, kernel_size=3, stride=1, padding=1),
                  #BasicConv(512, 512, kernel_size=3, stride=2, padding=1), \
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=1, padding=0),
                  BasicConv(256, 256, kernel_size=3, stride=1, padding=0)]
    elif size == 512:
        layers = [BasicConv(256 * 3, 512, kernel_size=3, stride=1, padding=1),
                  #BasicConv(512, 512, kernel_size=3, stride=2, padding=1), \
                  BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=4, padding=1, stride=1)]
    return layers

##
# Date: 11/20
# 512->256 
#
def pyramid_feature_extractor_3(size):
    if size == 300:
        layers = [BasicConv(128 * 3, 256, kernel_size=3, stride=1, padding=1),
                  #BasicConv(512, 512, kernel_size=3, stride=2, padding=1), \
                  #BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=1, padding=0),
                  BasicConv(256, 256, kernel_size=3, stride=1, padding=0)]
    elif size == 512:
        layers = [BasicConv(128 * 3, 256, kernel_size=3, stride=1, padding=1), # may 512->256
                  #BasicConv(512, 512, kernel_size=3, stride=2, padding=1), \
                  #BasicConv(512, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1),
                  BasicConv(256, 256, kernel_size=3, stride=2, padding=1), \
                  BasicConv(256, 256, kernel_size=4, padding=1, stride=1)]
    return layers

def multibox(fea_channels, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    assert len(fea_channels) == len(cfg)
    for i, fea_channel in enumerate(fea_channels):
        loc_layers += [nn.Conv2d(fea_channel, cfg[i] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(fea_channel, cfg[i] * num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)


extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}


mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}
#channels change based on FMs channels
#fea_channels = {
#    '300': [512, 512, 256, 256, 256, 256],
#    '512': [512, 512, 256, 256, 256, 256, 256]}
fea_channels = {
    '300': [512, 1024, 768, 768, 768, 768],
    '512': [512, 1024, 768, 768, 768, 768, 768]}

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FSSD(nn.Module):

    def __init__(self, num_classes, size): # add planes

        super(FSSD, self).__init__()
        self.num_classes = num_classes
        self.size = size
        # SSD network
        self.base =  get_vgg16_fms()
        self.extras = nn.ModuleList(add_extras(extras[str(self.size)], 1024))
        self.ft_module1 = nn.ModuleList(feature_transform_module_1(self.size))
        self.pyramid_ext1 = nn.ModuleList(pyramid_feature_extractor_1(self.size))
        self.ft_module2 = nn.ModuleList(feature_transform_module_2(self.size))
        self.pyramid_ext2 = nn.ModuleList(pyramid_feature_extractor_2(self.size))
        self.ft_module3 = nn.ModuleList(feature_transform_module_3(self.size))
        self.pyramid_ext3 = nn.ModuleList(pyramid_feature_extractor_3(self.size))
        self.fea_bn1 = nn.BatchNorm2d(256 * len(self.ft_module1), affine=True)
        self.fea_bn2 = nn.BatchNorm2d(256 * len(self.ft_module2), affine=True)
        self.fea_bn3 = nn.BatchNorm2d(128 * len(self.ft_module3), affine=True)


        head = multibox(fea_channels[str(self.size)], mbox[str(self.size)], self.num_classes)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax()

        # se for 300 size
        self.se1 = SELayer(512, 16)
        self.se2 = SELayer(1024, 16)
        self.se3 = SELayer(768, 16)
        self.se4 = SELayer(768, 16)
        self.se5 = SELayer(768, 16)
        self.se6 = SELayer(768, 16)
        # se for 512
        self.se7 = SELayer(768, 16)

        #self.cbam = CBAM(planes, 16)

    def get_pyramid_feature(self, x):
        source_fms = list()
        # conv4_3 and fc7
        source_fms += self.base(x)

        # conv6_2
        x = source_fms[-1]
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                source_fms.append(x)
        

        # first fusion module
        #assert len(source_fms) == len(self.ft_module1)
        transformed_features_1 = list()
        for k, v in enumerate(self.ft_module1):
            x = v(source_fms[k])
            transformed_features_1.append(x)
        concat_fea = torch.cat(transformed_features_1, 1)
        x = self.fea_bn1(concat_fea)
        
        # cbam on final
        #x = self.cbam2(x)

        # first pyramid feature generate
        pyramid_fea_1 = list()

        for k,v in enumerate(self.pyramid_ext1):
            x = v(x)
            pyramid_fea_1.append(x)       

        # 2nd fusion module
        #assert len(source_fms) == len(self.ft_module2)
        transformed_features_2 = list()
        for k, v in enumerate(self.ft_module2):
            x = v(source_fms[k+1])
            transformed_features_2.append(x)
        concat_fea = torch.cat(transformed_features_2, 1)
        x = self.fea_bn2(concat_fea)

        pyramid_fea_2 = list()

        for k,v in enumerate(self.pyramid_ext2):
            x = v(x)
            pyramid_fea_2.append(x)  

        # 3rd fusion module
        #assert len(source_fms) == len(self.ft_module2)
        transformed_features_3 = list()
        for k, v in enumerate(self.ft_module3):
            x = v(source_fms[k+2])
            transformed_features_3.append(x)
        concat_fea = torch.cat(transformed_features_3, 1)
        x = self.fea_bn3(concat_fea)

        pyramid_fea_3 = list()

        for k,v in enumerate(self.pyramid_ext3):
            x = v(x)
            pyramid_fea_3.append(x)  

        pyramid_fea_final = list()
        
        #ipdb.set_trace()
        
        # add cbam&se block latter
        

        # Date: 11/19
        # remove original features
        # Date: 11/20
        # add SE block
        ##
        #concat_fea = torch.cat([source_fms[0], pyramid_fea_1[0]], 1)
        pyramid_fea_final.append(self.se1(pyramid_fea_1[0]))
        #concat_fea = torch.cat([source_fms[1], pyramid_fea_1[1], pyramid_fea_2[0]], 1)
        concat_fea = torch.cat([pyramid_fea_1[1], pyramid_fea_2[0]], 1)
        pyramid_fea_final.append(self.se2(concat_fea))
        #concat_fea = torch.cat([source_fms[2], pyramid_fea_1[2], pyramid_fea_2[1], pyramid_fea_3[0]], 1)
        concat_fea = torch.cat([pyramid_fea_1[2], pyramid_fea_2[1], pyramid_fea_3[0]], 1)
        pyramid_fea_final.append(self.se3(concat_fea))
        #concat_fea = torch.cat([source_fms[3], pyramid_fea_1[3], pyramid_fea_2[2], pyramid_fea_3[1]], 1)
        concat_fea = torch.cat([pyramid_fea_1[3], pyramid_fea_2[2], pyramid_fea_3[1]], 1)
        pyramid_fea_final.append(self.se4(concat_fea))
        #concat_fea = torch.cat([source_fms[4], pyramid_fea_1[4], pyramid_fea_2[3], pyramid_fea_3[2]], 1)
        concat_fea = torch.cat([pyramid_fea_1[4], pyramid_fea_2[3], pyramid_fea_3[2]], 1)
        pyramid_fea_final.append(self.se5(concat_fea))
        #concat_fea = torch.cat([source_fms[5], pyramid_fea_1[5], pyramid_fea_2[4], pyramid_fea_3[3]], 1)
        concat_fea = torch.cat([pyramid_fea_1[5], pyramid_fea_2[4], pyramid_fea_3[3]], 1)
        pyramid_fea_final.append(self.se6(concat_fea))

        # for size 512
        concat_fea = torch.cat([pyramid_fea_1[6], pyramid_fea_2[5], pyramid_fea_3[4]], 1)
        pyramid_fea_final.append(self.se7(concat_fea))
        
        #ipdb.set_trace()
        return pyramid_fea_final



    def forward(self, x, test=False):
        loc = list()
        conf = list()

        pyramid_fea = self.get_pyramid_feature(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(pyramid_fea, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if test:
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def init_model(self, base_model_path):

        base_weights = torch.load(base_model_path)
        print('Loading base network...')
        self.base.layers.load_state_dict(base_weights)


        def xavier(param):
            init.xavier_uniform(param)


        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0
        print('Initializing weights...')
        self.extras.apply(weights_init)
        self.ft_module1.apply(weights_init)
        self.pyramid_ext1.apply(weights_init)
        self.ft_module2.apply(weights_init)
        self.pyramid_ext2.apply(weights_init)
        self.ft_module3.apply(weights_init)
        self.pyramid_ext3.apply(weights_init)
        self.loc.apply(weights_init)
        self.conf.apply(weights_init)


    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')




def build_net(size=300, num_classes=21):
    if size != 300 and size != 512:
        print("Error: Sorry only FSSD300 and FSSD512 is supported currently!")
        return

    return FSSD(num_classes=num_classes,size=size)