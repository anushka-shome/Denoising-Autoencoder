"""=================================================================================================
https://www.datacamp.com/tutorial/introduction-to-autoencoders

Michael Gribskov     28 March 2025
================================================================================================="""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mpl_toolkits.mplot3d import Axes3D


class Autoencoder(nn.Module):

    def __init__(self, input_size, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU()
            )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_size),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# --------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    df = pd.read_csv("data/weatherAUS.csv")
    df = df.drop(['Date', 'Location', 'WindDir9am',
                  'WindGustDir', 'WindDir3pm'], axis=1)
    df = df.dropna(how='any')
    df.loc[df['RainToday'] == 'No', 'RainToday'] = 0
    df.loc[df['RainToday'] == 'Yes', 'RainToday'] = 1
    df.head()

    X, Y = df.drop('RainTomorrow', axis=1, inplace=False), df[['RainTomorrow']]
    # Normalizing Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Converting to PyTorch tensor
    X_tensor = torch.FloatTensor(X_scaled)

    # Converting string labels to numerical labels
    label_encoder = LabelEncoder()
    Y_numerical = label_encoder.fit_transform(Y.values.ravel())

    # Setting random seed for reproducibility
    # torch.manual_seed(42)

    input_size = X.shape[1]  # Number of input features
    encoding_dim = 3  # Desired number of output dimensions
    model = Autoencoder(input_size, encoding_dim)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # Training the autoencoder
    num_epochs = 2000
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, X_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss for each epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Encoding the data using the trained autoencoder
    encoded_data = model.encoder(X_tensor).detach().numpy()

    # Plotting the encoded data in 3D space
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # ax.axes.set_xlim3d(left=-10, right=50)
    # ax.axes.set_ylim3d(bottom=0, top=20)
    # ax.axes.set_zlim3d(bottom=-10, top=50)
    scatter = ax.scatter(encoded_data[:, 0], encoded_data[:, 1],
                         encoded_data[:, 2], c=Y_numerical, cmap='viridis')

    # Mapping numerical labels back to original string labels for the legend
    labels = label_encoder.inverse_transform(np.unique(Y_numerical))
    legend_labels = {num: label for num, label in zip(np.unique(Y_numerical),
                                                      labels)}

    # Creating a custom legend with original string labels
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=scatter.to_rgba(num),
                          markersize=10,
                          label=legend_labels[num]) for num in np.unique(Y_numerical)]
    ax.legend(handles=handles, title="Raining Tomorrow?")

    # Adjusting the layout to provide more space for labels
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Manually adding z-axis label for better visibility
    ax.text2D(0.05, 0.95, 'Encoded Dimension 3', transform=ax.transAxes,
              fontsize=12, color='black')

    ax.set_xlabel('Encoded Dimension 1')
    ax.set_ylabel('Encoded Dimension 2')
    ax.set_title('Autoencoder Dimensionality Reduction')

    plt.tight_layout()

    plt.savefig('Rain_Prediction_Autoencoder.png')

    plt.show()

    # exit(0)
