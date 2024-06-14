import torch.nn as nn

class FineGrainedPredictor(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FineGrainedPredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes + 1)  
        )

    def forward(self, x):
        return self.predictor(x)
