import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

class Network(nn.Module):
    def __init__(self, input_shape: int):
        super().__init__()
        self.encode1 = nn.Linear(input_shape, 1000)
        self.encode2 = nn.Linear(1000, 500)
        self.encode3 = nn.Linear(500, 250)
        self.encode4 = nn.Linear(250, 100)
        self.encode5 = nn.Linear(100, 50)

        self.decode1 = nn.Linear(50, 100)
        self.decode2 = nn.Linear(100, 250)
        self.decode3 = nn.Linear(250, 500)
        self.decode4 = nn.Linear(500, 1000)
        self.decode5 = nn.Linear(1000, input_shape)
    def encode(self, x: torch.Tensor):
        x = F.relu(self.encode1(x))
        x = F.relu(self.encode2(x))
        x = F.relu(self.encode3(x))
        x = F.relu(self.encode4(x))
        x = F.relu(self.encode5(x))
        return x
    def decode(self, x: torch.Tensor):
        x = F.relu(self.decode1(x))
        x = F.relu(self.decode2(x))
        x = F.relu(self.decode3(x))
        x = F.relu(self.decode4(x))
        x = F.relu(self.decode5(x))
        return x
    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        x = self.decode(x)
        return x


def train_model(data: torch.Tensor, input_size: int, batch_size=128, epochs=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # Check if running on GPU or CPU
    net = Network(input_size).to(device)
    #optimizer = optim.Adagrad(net.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    dataset = torch.utils.data.TensorDataset(data, data) #input and target are the same
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Before epochs")
    for epoch in range(epochs):
        print(f"epoch: {epoch}")
        epoch_loss = 0
        batch_num = 1
        for batch in dataloader:
            #print(f"batch: {batch_num}")
            batch_num += 1
            batch = batch[0].to(device)
            net.zero_grad()

            # Pass batch through 
            output = net(batch)

            # Get Loss + Backprop
            loss = loss_fn(output, batch) # 
            #losses.append(loss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss/len(dataloader))
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {losses[-1]}")
            torch.save(net.state_dict(), "model2.pth")
    
    torch.save(net.state_dict(), "model2.pth")
    return net, losses

def main():
    df = pd.read_csv("binary_data.csv", header=None)
    data = torch.tensor(df.values, dtype=torch.float32)
    input_size = data.shape[1]
    net, losses = train_model(data, input_size)
    print(net)
    print(losses)

if __name__ == "__main__":
    main()
