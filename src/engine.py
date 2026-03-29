import numpy as np
import torch
import torch.nn as nn

def train_model(model, train_loader, epochs, lr, wd, cuda):

    print("🚀 Start Training...")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    model.train()
    for e in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            if cuda:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            opt.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (e+1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {e+1}/{epochs} | Loss: {avg_loss:.6f}')

def predict(model, test_loader, cuda):

    model.eval()
    predictions = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            if cuda:
                x_batch = x_batch.cuda()
            output = model(x_batch)
            predictions.append(output.cpu().numpy())
    return np.concatenate(predictions)

def predict_next_day(model, scaler, features, seq_len, device):

    last_window_raw = features[-seq_len:]
    last_window_scaled = scaler.transform(last_window_raw)
    
    input_tensor = torch.FloatTensor(last_window_scaled).unsqueeze(0)
    if device:
        input_tensor = input_tensor.cuda()
    
    model.eval()
    with torch.no_grad():
        pred_pct = model(input_tensor).item()
        
    return pred_pct