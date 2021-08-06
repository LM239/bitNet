import torch
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, conv3x3

class ResNet(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS

        self.model = models.resnet34(pretrained=True, progress=True)

        self.layer5 = BasicBlock(inplanes=512, planes=256, stride=2, downsample=conv3x3(512, 256, 2))
        self.layer6 = BasicBlock(inplanes=256, planes=256, stride=2, downsample=conv3x3(256, 256, 2))
        self.layer7 = BasicBlock(inplanes=256, planes=256, stride=3, downsample=conv3x3(256, 256, 3))
        
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)
        
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        
        out_features = [x2, x3, x4, x5, x6, x7]
        
        return tuple(out_features)

