import torch
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super(FeatureExtractor, self).__init__()
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.output_dim = 2048
        elif model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.output_dim = 512
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=pretrained)
            self.model = nn.Sequential(*list(self.model.features.children())[:-1])
            self.output_dim = 4096
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")

    def forward(self, x):
        features = self.model(x)
        if isinstance(features, torch.Tensor):
            features = features.view(features.size(0), -1)
        return features

    def get_output_dim(self):
        return self.output_dim
