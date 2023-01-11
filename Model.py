import torch

from transformers import AutoFeatureExtractor, Swinv2Model, Swinv2ForImageClassification


class SwinDenoiser(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.model = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.head = torch.nn.Sequential(torch.nn.Conv2d(64, 3, 3))

    def forward(self, x):
        embedded_features = self.feature_extractor(x, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**embedded_features)

        return self.head(outputs.last_hidden_state)