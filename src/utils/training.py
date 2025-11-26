import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

def train_autoencoder(model, data, epochs, batch_size, device='cpu'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    # Prepare data
    if isinstance(data, list):
        # Check if list of tensors or numpy arrays
        if len(data) > 0 and isinstance(data[0], (np.ndarray, torch.Tensor)):
             pass # assumed handled
    
    # Convert to tensor dataset
    if isinstance(data, np.ndarray):
        tensor_data = torch.from_numpy(data).float()
    elif isinstance(data, torch.Tensor):
        tensor_data = data.float()
    elif isinstance(data, datasets.MNIST): # This wont work directly
         pass 
    else:
        # Assume standard list of tensors
        tensor_data = torch.stack(data).float()
        
    # Flatten input for MLP AE
    flat_data = tensor_data.view(tensor_data.size(0), -1)
    
    dataset = TensorDataset(flat_data, flat_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training Autoencoder on {len(data)} samples for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            encoded, decoded = model(inputs)
            loss = criterion(decoded, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")

    return model
