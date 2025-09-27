import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# 3-layer encoder-decoder network (autoencoder)
class EncoderDecoderNet(nn.Module):
    def __init__(self):
        super(EncoderDecoderNet, self).__init__()
        # Encoder: 3 inputs -> 16 -> 8 -> 5
        self.encoder = nn.Sequential(
            nn.Linear(3, 16,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 5),
            nn.ReLU()
        )
        # Decoder: 4 -> 8 -> 16 -> 5 outputs
        self.decoder = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def generate_data():
    F1 = (-10000, 10000)
    F2 = (-5000, 5000)
    F3 = (-5000, 5000)
    alpha2 = (-np.pi, np.pi)
    alpha3 = (-np.pi, np.pi)

    dataset_size = 1*10^6
    # train_size = .8
    # val_size = .2
    # batch_size = 128
    # learning_rate = 0.001
    # num_epochs = 1000
    # hidden_size = 128
    # num_layers = 2
    # dropout = 0.2
    # input_size = 5
    # output_size = 3

    l1 = -14
    l2 = 14.5
    l3 = -2.7 
    l4 = 2.7


    tau = [
            [0., np.cos(alpha2), np.cos(alpha3)],
            [1., np.sin(alpha2), np.sin(alpha3)],
            [l2, l1*np.sin(alpha2)-l3*(np.cos(alpha2)),l1*np.sin(alpha3)-l4*np.cos(alpha3)]
            
        ]

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
        # print(res)
        
        F_Data = np.hstack(([F1_r, F2_r, F3_r,alpha2_r,alpha3_r],res),dtype=float)
        F_Train.append(F_Data)

    end = time.time()
    print(f"Execution time: {end - start} seconds")
    print("Records Generated: {}".format(len(F_Train)))
    F_Train = np.array(F_Train)
    return F_Train


# Example training loop for EncoderDecoderNet
def train_model(model, train_loader, num_epochs=100000, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    best_loss_epoch = -1
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_loss_epoch = epoch
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                if epoch > best_loss_epoch+200:
                    print(f"Early stopping at epoch {epoch}")
                    return
                
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# Example usage
if __name__ == "__main__":
    
    sample_data = generate_data()
    sample_data_tensor = torch.from_numpy(sample_data)
    sample_data_tensor=sample_data_tensor.to(torch.float32)
    # Example: create dummy data and DataLoader
    # inputs = torch.randn(1000, 3)      # 1000 samples, 3 features each
    # targets = torch.randn(1000, 5)     # 1000 samples, 5 targets each

    inputs=sample_data_tensor[:,:3]
    
    targets=sample_data_tensor[:,3:]
    
    targets=sample_data_tensor[:,:3]
    
    # Suppose inputs and targets are your full dataset tensors
    from sklearn.model_selection import train_test_split

    # Split: 80% train, 20% test
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(
        inputs, targets, test_size=0.2, random_state=42
    )

    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
    test_dataset = torch.utils.data.TensorDataset(test_inputs, test_targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    model = EncoderDecoderNet()
    train_model(model, train_loader)

    output = model(test_dataset[0][0])  # Use first 2 samples for demonstration
    print(output.device)
    print("Input:", test_dataset[0][0])
    print("Decoded Output:", output)
    print("Actual Output:", test_dataset[0][1])
    print("Output shape:", output.shape)  # Should be [2, 5]
