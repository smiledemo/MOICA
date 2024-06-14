import torch.nn as nn

class DomainClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
