import torch
from tqdm import tqdm

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(iter(test_loader)):
            features = features.float().to(device)
            
            probs = model(features)

            probs = probs.cpu().detach().numpy()
            predictions += probs.tolist()
    return predictions
