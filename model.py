import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colormaps
from sklearn.manifold import TSNE
from types_dict import get_array_key, get_array_entry
import numpy as np
import plotly.express as px
from noise_vectors import get_array
import random

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

class Network2(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super().__init__()
        self.encoder = nn.Sequential(
            #nn.Linear(input_shape, 1000),
            #nn.ReLU(),
            #nn.Linear(1000, 500),
            #nn.LeakyReLU(),
            #nn.Linear(500, 250),
            #nn.LeakyReLU(),
            #nn.Linear(250, 100),
            #nn.LeakyReLU(),
            #nn.Linear(100, 50),
            #nn.LeakyReLU()
            nn.Linear(input_shape, 32), #3, 2, 6
            nn.LeakyReLU(),
            nn.Linear(32, 6),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.LeakyReLU(),
            nn.Linear(32, output_shape) # No ReLU here for final output
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_model(data: torch.Tensor, input_size: int, batch_size=128, epochs=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # Check if running on GPU or CPU
    net = Network2(input_size).to(device)
    #optimizer = optim.Adagrad(net.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    dataset = torch.utils.data.TensorDataset(data, data) #input and target are the same
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data = data.to(device)
    output = net(data)
    loss_init = loss_fn(output, data).item()
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
            loss_percentage = 100 * (losses[-1]/loss_init)
            with open("leaky_seq_loss.txt", "a") as f:
                print(f"Epoch {epoch}: Loss = {losses[-1]} Loss % = {loss_percentage:.4f}%", file=f)
            torch.save(net.state_dict(), "leaky_seq_model.pth")
    torch.save(net.state_dict(), "leaky_seq_model.pth")
    return net, losses

def train_model_noise(data: torch.Tensor, target: torch.Tensor, input_size: int, final_size: int, batch_size=128, epochs=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # Check if running on GPU or CPU
    net = Network2(input_size, final_size).to(device)
    #optimizer = optim.Adagrad(net.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction='sum') # Try L1
    #loss_fn = nn.MSELoss()
    #loss_fn = nn.MSELoss(reduction='sum')
    losses = []
    data_inp = data.unsqueeze(0)
    target_inp = target.unsqueeze(0)
    dataset = torch.utils.data.TensorDataset(data_inp, target_inp) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data = data.to(device)
    data_inp = data_inp.to(device)
    target_inp = target_inp.to(device)
    print(f"Target: {target_inp.shape}")
    target = target.to(device)
    output = net(data_inp)
    print(f"Output shape: {output.shape}")
    loss_init = loss_fn(output, target_inp).item()
    with open("noise_max10.pkl", "rb") as f:
        noise = pickle.load(f)
    print("Before epochs") # Add noise each epoch
    for epoch in range(epochs):
        print(f"epoch: {epoch}")
        new_rows = data.clone()
        for i in range(173):
            for j in range(1): # somewhere between 1 and 10
                row_index = random.randint(0, 9) + i * 10
                dict_index = random.randint(0, 100)
                new_rows[row_index] = torch.tensor(noise[i][dict_index], dtype=torch.float32) #change index to row_index for any < 10
        new_data = new_rows.to(device)
        dataset = torch.utils.data.TensorDataset(new_data, target.to(device))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        batch_num = 1
        for batch, batch_targ in dataloader:
            #print(f"batch: {batch_num}")
            batch_num += 1
            batch = batch.to(device)
            batch_targ = batch_targ.to(device)
            net.zero_grad()

            # Pass batch through 
            output = net(batch)

            # Get Loss + Backprop
            loss = loss_fn(output, batch_targ) # 
            #losses.append(loss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss/len(dataloader))
        if epoch % 10 == 0:
            loss_percentage = 100 * (losses[-1]/loss_init)
            with open("noise_loss_r_ch_32_6.txt", "a") as f:
                print(f"Epoch {epoch}: Loss = {losses[-1]} Loss % = {loss_percentage:.4f}%", file=f)
            torch.save(net.state_dict(), "leaky_noise.pth")
    torch.save(net.state_dict(), "leaky_noise_samedim.pth")
    #return net, losses
    output = net(data)
    output_binary = (torch.sigmoid(output) > 0.5).float()
    output_list = output_binary.detach().cpu().numpy().tolist()
    pd.DataFrame(output_list).to_csv("op_noise_col.csv", index=False, header=False)
    print(output)
    return net.encoder(data).detach().cpu().numpy()

def main():
    df = pd.read_csv("condensed_data10.csv", header=None) # Per epoch change the noise
    target = torch.tensor(df.values, dtype=torch.float32) # 
    dft = pd.read_csv("rows_noise_c_max10.csv", header=None)
    data = torch.tensor(dft.values, dtype=torch.float32)
    input_size = data.shape[1]
    final_size = target.shape[1]
    print(input_size)
    print(final_size)
    print(f"Input size: {input_size}")
    print(f"Final size: {final_size}")
    encoded_data = train_model_noise(data, target, input_size, final_size)
    print(encoded_data.shape)
    print(encoded_data.T.shape)
    #print(net)
    #print(losses)
    #pca = PCA(n_components=3)
    tsne = TSNE(n_components=3, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    p = tsne.fit_transform(encoded_data)

    with open("classes_curated.pkl", "rb") as f:
        class_dict = pickle.load(f)
    #labels = [class_dict[i] for i in range(len(class_dict))]
    labels = [class_dict[i // 10] for i in range(len(p))]

    unique_classes = list(set(labels))
    colors = colormaps['tab20']
    class_to_color = {cls: colors(i) for i, cls in enumerate(unique_classes)}
    point_colors = [class_to_color[label] for label in labels]

    df_plot = pd.DataFrame(p, columns=["x", "y", "z"])
    df_plot["label"] = labels

    fig = px.scatter_3d(df_plot, x="x", y="y", z="z", color="label", title="TSNE of Latent Space", width=900, height=700)

    fig.update_traces(marker=dict(size=4))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

    fig.show()
    fig.write_html("noise_tsne_r_ch_32_6.html") #hierarchal clustering
    # add noise in the columns

    '''
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=point_colors, s=50, edgecolor='k')
    ax.set_title("PCA of Latent Space")
    ax.set_xlabel("1")
    ax.set_ylabel("2")
    ax.set_zlabel("3")
    for cls in unique_classes:
        ax.scatter([], [], [], c=[class_to_color[cls]], label=cls)
    ax.legend(loc='best', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()
    plt.savefig("noise_pca.png")
    '''

if __name__ == "__main__":
    main()
