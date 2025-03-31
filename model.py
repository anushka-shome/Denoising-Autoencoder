import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd


class Network(nn.Module):

    def __init__(self, input_shape: int):
        super().__init__()
        self.encode1 = nn.Linear(input_shape, 10000)
        self.encode2 = nn.Linear(10000, 1000)
        # self.encode3 = nn.Linear(500, 250)
        # self.encode4 = nn.Linear(250, 100)
        # self.encode5 = nn.Linear(100, 50)
        self.encode5 = nn.Linear(1000, 100)

        self.decode1 = nn.Linear(100, 1000)
        # self.decode1 = nn.Linear(50, 100)
        # self.decode2 = nn.Linear(100, 250)
        # self.decode3 = nn.Linear(250, 500)
        self.decode4 = nn.Linear(1000, 10000)
        self.decode5 = nn.Linear(10000, input_shape)

    def encode(self, x: torch.Tensor):
        x = F.relu(self.encode1(x))
        x = F.relu(self.encode2(x))
        # x = F.relu(self.encode3(x))
        # x = F.relu(self.encode4(x))
        x = F.relu(self.encode5(x))
        return x

    def decode(self, x: torch.Tensor):
        x = F.relu(self.decode1(x))
        # x = F.relu(self.decode2(x))
        # x = F.relu(self.decode3(x))
        x = F.relu(self.decode4(x))
        x = F.relu(self.decode5(x))
        return x

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        x = self.decode(x)
        return x


class Autoencoder(nn.Module):
    """=============================================================================================
    Simple autoencoder from datacamp example
    ============================================================================================="""

    def __init__(self, input_size, hlayer_size, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hlayer_size),
            nn.LeakyReLU(),
            nn.Linear(hlayer_size, encoding_dim),
            nn.LeakyReLU()
            )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hlayer_size),
            nn.LeakyReLU(),
            nn.Linear(hlayer_size, input_size),
            nn.LeakyReLU()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_model(data: torch.Tensor, input_size: int, batch_size=16, epochs=1000):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")  # Check if running on GPU or CPU

    net = Network(input_size).to(device)
    # optimizer = optim.Adagrad(net.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
    losses = []
    dataset = torch.utils.data.TensorDataset(data, data)  # input and target are the same
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Before epochs")
    loss_min = 100000000
    for epoch in range(epochs):
        print(f"epoch: {epoch}", end='\t')
        epoch_loss = 0
        batch_num = 1
        for batch in dataloader:
            # print(f"batch: {batch_num}")
            batch_num += 1
            batch = batch[0].to(device)
            net.zero_grad()

            # Pass batch through 
            output = net(batch)

            # Get Loss + Backprop
            loss = loss_fn(output, batch)
            loss_min = min(loss_min, loss)
            losses.append(loss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(dataloader))
        if epoch % 1 == 0:
            print(f"Loss: {loss:8.1f}\t   min: {loss_min:8.1f}")
            # torch.save(net.state_dict(), "model2.pth")

    torch.save(net.state_dict(), "model2.pth")
    return net, losses


def train_model_nobatch(data: torch.Tensor, input_size: int, epochs=500):
    print(f"Setting up for training: epochs = {epochs}")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")  # Check if running on GPU or CPU

    net = Autoencoder(input_size, 4,4)
    # optimizer = optim.Adagrad(net.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = optim.Adam(net.parameters(), lr=5e-2)
    # loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
    loss_fn = nn.MSELoss(reduction='sum')
    dataset = torch.utils.data.TensorDataset(data, data)  # input and target are the same
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)

    print(f'Begin training')
    output = net(data)
    loss_init = loss_fn(output, data)
    loss_min = loss_init
    loss_min_step = 0
    for epoch in range(epochs):
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)
        print(f"epoch: {epoch}", end='\t')
        epoch_loss = 0

        # forward
        output = net(data)
        loss = loss_fn(output, data)

        # minimum loss and epoch when it occurred
        if loss < loss_min:
            loss_min = loss
            loss_min_step = epoch

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss:8.1f}\t   min: {loss_min:8.1f}/{loss_min_step}\t{100 * loss_min / loss_init:.3f}%")

    out = output.detach().numpy()
    return net.encoder(data).detach().numpy()


def main():
    df = pd.read_csv("data/binary_data.csv", header=None).T
    data = torch.tensor(df.values, dtype=torch.float32)
    input_size = data.shape[1]
    hlayer_size = 4
    encoded_data = train_model_nobatch(data, input_size, epochs=2000)
    for i in range(len(encoded_data)):
        print(f'{i}',end='')
        for value in encoded_data[i]:
            print(f'\t{value:.4f}', end='')
        print()
    # print(net)
    # print(losses)


if __name__ == "__main__":
    main()
