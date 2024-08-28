
import torch
import torchvision.models as models


class Resnet(torch.nn.Module): # McMahan et al., 2016; 1,663,370 parameters
    def __init__(self,  num_classes):
        super(Resnet, self).__init__()
        self.num_classes = num_classes
    
        self.features = models.resnet18(models.ResNet18_Weights.DEFAULT)
        self.classifier = torch.nn.Linear(self.features.fc.in_features, num_classes)
        self.features.fc=torch.nn.Identity()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x