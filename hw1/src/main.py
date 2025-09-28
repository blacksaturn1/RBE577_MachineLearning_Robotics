import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3-layer encoder-decoder network (autoencoder)
class EncoderDecoderNet(nn.Module):
    def __init__(self):
        super(EncoderDecoderNet, self).__init__()
        # Encoder: 3 inputs -> 16 -> 8 -> 5
        self.encoded = None
        self.encoder = nn.Sequential(
            nn.Linear(3, 16,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 5),
            nn.ReLU()
        )
        # Decoder: 5 -> 8 -> 16 -> 3 outputs
        self.decoder = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def getEncoded(self):
        return self.encoded
    
    def forward(self, x):
        self.encoded = self.encoder(x)
        decoded = self.decoder(self.encoded)
        return decoded

F1 = (-10000, 10000)
F2 = (-5000, 5000)
F3 = (-5000, 5000)
alpha2 = (-np.pi, np.pi)
alpha3 = (-np.pi, np.pi)

def generate_data():
    
    dataset_size = 1*10**2
    l1 = -14
    l2 = 14.5
    l3 = -2.7 
    l4 = 2.7
    F_Train = []
    # time the data generation
    start = time.time()
    for i in range(dataset_size):
        F1_r = random.normalvariate(F1[0], F1[1])
        F2_r = random.normalvariate(F2[0], F2[1])
        F3_r = random.normalvariate(F3[0], F3[1])
        alpha2_r = random.normalvariate(alpha2[0], alpha2[1])
        alpha3_r = random.normalvariate(alpha3[0], alpha3[1])

        tau = [
                [0., np.cos(alpha2_r), np.cos(alpha3_r)],
                [1., np.sin(alpha2_r), np.sin(alpha3_r)],
                [l2, l1*np.sin(alpha2_r)-l3*(np.cos(alpha2_r)),l1*np.sin(alpha3_r)-l4*np.cos(alpha3_r)]
                
            ]
        tau = np.array(tau,dtype=float)
        F = np.array([F1_r, F2_r, F3_r],dtype=float)
        
        res = np.matmul(tau, F)
        
        F_Data = np.hstack(([F1_r, F2_r, F3_r,alpha2_r,alpha3_r],res),dtype=float)
        F_Train.append(F_Data)

    end = time.time()
    print(f"Execution time: {end - start} seconds")
    print("Records Generated: {}".format(len(F_Train)))
    F_Train = np.array(F_Train)
    # F_Train = torch.from_numpy(F_Train).to(torch.float32).to(device)
    return F_Train

def rateChangeLoss(u_actual):
    return 0

def custom_loss(model, output, target):
    L1 = nn.MSELoss()(output, target)
    # Add your custom logic, e.g., regularization or extra terms
    u_actual=model.getEncoded()
    # Thruster Command Magnitude Limits Loss
    L2 = thrusterCommandMagnitude(u_actual)
    L3 = rateChangeLoss(u_actual)
    return L1+L2+L3

def thrusterCommandMagnitude(u_actual):
    result = torch.clamp(abs(u_actual[:,0])-F1[1], min=0)
    L2 = result.mean() 
    result = torch.clamp(abs(u_actual[:,1])-F2[1], min=0)
    L2 += result.mean()
    result = torch.clamp(abs(u_actual[:,2])-F3[1], min=0)
    L2 += result.mean()
    result = torch.clamp(abs(u_actual[:,3])-alpha2[1], min=0)
    L2 += result.mean()
    result = torch.clamp(abs(u_actual[:,4])-alpha3[1], min=0)
    L2 += result.mean()
    return L2

# Example training loop for EncoderDecoderNet
def train_model(model, train_loader, test_loader, num_epochs=1000, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    best_loss_epoch = -1
    train_losses = []
    test_losses = []
    lossCounter = 0
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        train_count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = custom_loss(model,outputs, targets)
            running_train_loss += loss.item() * inputs.size(0)
            train_count += inputs.size(0)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_loss_epoch = epoch
                torch.save(model.state_dict(), 'best_model.pth')
            # else:
            #     if epoch > best_loss_epoch+200:
            #         print(f"Early stopping at epoch {epoch}")
            #         break
            loss.backward()
            optimizer.step()
        avg_train_loss = running_train_loss / train_count
        train_losses.append(avg_train_loss)

        # Evaluate test loss
        model.eval()
        running_test_loss = 0.0
        test_count = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                test_loss = custom_loss(model,outputs, targets)
                running_test_loss += test_loss.item() * inputs.size(0)
                test_count += inputs.size(0)
        avg_test_loss = running_test_loss / test_count
        test_losses.append(avg_test_loss)

        if avg_test_loss < avg_train_loss:
            lossCounter += 1
        if lossCounter > 200:
            print(f"Early stopping at epoch {epoch}")
            break
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss}")

    # Plot losses after training
    print(f"Best Loss: {best_loss} at epoch {best_loss_epoch}")
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.show()

def evaluate_loss(model, data_loader):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            #loss = criterion(outputs, targets)
            loss = custom_loss(model,outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)
    avg_loss = total_loss / count
    return avg_loss

# Example usage
if __name__ == "__main__":
    
    sample_data = generate_data()
    sample_data_tensor = torch.from_numpy(sample_data).to(torch.float32).to(device)
    
    inputs = sample_data_tensor[:, :3]
    targets=sample_data_tensor[:,:3]
    
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(
        inputs, targets, test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
    test_dataset = torch.utils.data.TensorDataset(test_inputs, test_targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    model = EncoderDecoderNet().to(device)
    train_model(model, train_loader, test_loader)
    data = torch.tensor(test_dataset[0][0],dtype=torch.float32).to(device)
    output = model(data)
    print(output.device)
    print("Input:", test_dataset[0][0])
    print("Decoded Output:", output)
    print("Actual Output:", test_dataset[0][1])
    print("Output shape:", output.shape)