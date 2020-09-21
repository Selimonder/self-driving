import torch,sys
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18
from typing import Dict
sys.path.append('src')
from effnet import *

class LyftModel(nn.Module):
    
    def __init__(self, cfg: Dict):
        super().__init__()
        
        self.backbone = resnet18(pretrained=True, progress=True)
        
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )
        
        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        backbone_out_features = 512

        # X, Y coords for the future positions (output shape: Bx50x2)
        num_targets = 2 * cfg["model_params"]["future_num_frames"]

        # You can add more layers here.
        self.head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )

        self.logit = nn.Linear(4096, out_features=num_targets)
        
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.head(x)
        x = self.logit(x)
        
        return x
    

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

class LyftEffnet(nn.Module):
    
    def __init__(self, cfg: Dict):
        super().__init__()

        # X, Y coords for the future positions (output shape: Bx50x2)
        num_targets = 2 * cfg["model_params"]["future_num_frames"]
        
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        ##

        self.backbone = EfficientNet.from_pretrained('efficientnet-b1', in_channels=num_in_channels)
        backbone_out_features = self.backbone._fc.weight.shape[1]

        self.gru = torch.nn.GRU(input_size=backbone_out_features, hidden_size=backbone_out_features, 
                        num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(backbone_out_features * 2, num_targets)
        self.fc2 = nn.Linear(num_targets, num_targets)

    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc1)

        
    def forward(self, x):

        x = self.backbone(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)        
        x = torch.mean(x, dim=3)
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        (x, _) = self.gru(x)
        x = F.relu_(self.fc1(x)).mean(dim=1)
        x = F.relu_(self.fc2(x))
        #x = x.transpose(1, 2)
        return x