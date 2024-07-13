import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, ASTModel, AutoProcessor
from .config import Config

class MLP(nn.Module):
    def __init__(self, model_name=Config.BACKBONE_NAME, input_dim=768, output_dim=Config.N_CLASSES):
        super(MLP, self).__init__()
        self.backbone_model = ASTModel.from_pretrained(model_name).to("cuda")
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

    def backbone(self, input_audio):    
        with torch.no_grad():
            outputs = self.backbone_model(input_audio)
            pooled_tensor = outputs.pooler_output
        
        return pooled_tensor

    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)  # 여러 클래스 분류라면 softmax를 사용합니다.
        return x