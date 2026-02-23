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
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, rand_score, normalized_mutual_info_score
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import sys

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
            nn.Linear(input_shape, 8), #3, 2, 6 8th of the size of the total vector
            nn.LeakyReLU(),
            nn.Linear(8, 3), # full parameter sweep
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.LeakyReLU(),
            nn.Linear(8, output_shape) # No ReLU here for final output
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

def train_model_noise(data: torch.Tensor, target: torch.Tensor, input_size: int, final_size: int, batch_size=282, epochs=10000): #10,000 epochs #batch_size=1410/5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # Check if running on GPU or CPU
    net = Network2(input_size, final_size).to(device)
    #optimizer = optim.Adagrad(net.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    #loss_fn = nn.BCEWithLogitsLoss(reduction='sum') # Try L1
    #loss_fn = nn.MSELoss()
    loss_fn = nn.MSELoss(reduction='sum') #changed this!
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
        '''
        for i in range(173):
            for j in range(1): # somewhere between 1 and 10
                row_index = random.randint(0, 9) + i * 10
                dict_index = random.randint(0, 100)
                new_rows[row_index] = torch.tensor(noise[i][dict_index], dtype=torch.float32) #change index to row_index for any < 10
        '''

        #df = pd.read_csv("condensed_predicted_trimmed_10.csv", header=None)
        '''
        df = pd.read_csv("condensed_predicted_trimmed_10.csv", header=None)
        #df_io = pd.read_csv("motif_io.csv", header=None)
        for i in range(1410):
          #i = random.randint(0, 9) + p * 10
          col_indices = [j for j,x in enumerate(df.iloc[i]) if x == 1]
          col_filter = [j for j,x in enumerate(new_rows[i]) if x >= 1]
          col_indicies = [j for j in col_indices if j not in col_filter]
          #i_edges = new_rows[i][len(new_rows[i])-2]
          #o_edges = new_rows[i][len(new_rows[i])-1]
          #print(len(col_indices))
          #col_num = random.randint(0, len(col_indices)) # 10% of the number of curated motifs or 20% for cross-validation (10, 20, 30, 40)
          col_num = (int)(0.10 * len(col_filter)) #fixed number
          col_num = 5
          for t in range(col_num):
            if len(col_indices) < 1:
              break
            col_choose = random.randint(0, len(col_indices) - 1)
            col_choose = col_indices[col_choose]
            new_rows[i, col_choose] = 1.0
            #new_rows[i, len(new_rows[i])-2] += df_io.iloc[0, col_choose]
            #new_rows[i, len(new_rows[i])-1] += df_io.iloc[1, col_choose]
            col_indices.remove(col_choose)

        new_data = new_rows.to(device)
        dataset = torch.utils.data.TensorDataset(new_data, target.to(device))
        '''
        dataset = torch.utils.data.TensorDataset(data, target.to(device))
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
            with open(f"loss_MSE_10e_pred_full_8_3.txt", "a") as f:
                print(f"Epoch {epoch}: Loss = {losses[-1]} Loss % = {loss_percentage:.4f}%", file=f)
            torch.save(net.state_dict(), "leaky_noise.pth")
    torch.save(net.state_dict(), "leaky_noise_samedim.pth")
    #return net, losses
    output = net(data)
    output_binary = (torch.sigmoid(output) > 0.5).float()
    output_list = output_binary.detach().cpu().numpy().tolist()
    pd.DataFrame(output_list).to_csv("op_noise_col.csv", index=False, header=False)
    print(output)
    W1 = net.encoder[0].weight.detach().cpu()      # (8, 1625)
    feat_strength = torch.norm(W1, dim=0).numpy()  # (1625,)

    topk = np.argsort(-feat_strength)[:30]         # top 30 features

    df_top = pd.DataFrame({
    "feature_idx": topk,
    "strength_L2": feat_strength[topk],
    })
    print(df_top)
    df_top.to_csv("top_features_pred_full_8_3.csv", index=False, header=False)
    return net.encoder(data).detach().cpu().numpy()

def sil_score(array, labels):
    x = np.array(array)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    n = x.shape[0]
    scores = []
    for i in range(n):
        xi = x[i].reshape(1, -1)
        cluster = labels[i]
        #a(i)
        same_cluster_mask = labels == cluster
        same_cluster_points = x[same_cluster_mask]
        if len(same_cluster_points) > 1:
            same_dists = cdist(xi, same_cluster_points)[0]
            ai = np.mean(same_dists[same_dists > 0])
        else:
            print("ai is 0")
            ai = 0
        #b(i)
        dists = []
        for c in unique:
            if c == cluster:
                continue
            other_points = x[labels == c]
            if other_points.ndim == 1:
                other_points = np.atleast_2d(other_points)
            mean = np.mean(cdist(xi, other_points))
            dists.append(mean)
        bi = np.min(dists)
        if max(ai, bi) != 0:
            si = (bi - ai)/max(ai, bi)
        else:
            si = 0
        scores.append(si)
    final_score = np.mean(np.array(scores))
    return final_score

def dbscan_sweep(X, y_true, eps_list, min_samples_list):
    rows = []
    for ms in min_samples_list:
        for eps in eps_list:
            db = DBSCAN(eps=eps, min_samples=ms)
            y_db = db.fit_predict(X)

            clusters = set(y_db)
            n_clusters = len([c for c in clusters if c != -1])
            noise_frac = float(np.mean(y_db == -1))

            ari = adjusted_rand_score(y_true, y_db)
            nmi = normalized_mutual_info_score(y_true, y_db)

            '''
            sil = np.nan
            if n_clusters >= 2 and n_clusters <= 70: # CHANGE TO 140
                mask = (y_db != -1)
                labels_kept = y_db[mask]
                n_labels = len(np.unique(labels_kept))
                if len(set(y_db[mask])) >= 2 and mask.sum() > 3 and 2 <= n_labels <= (mask.sum() - 1):
                    sil = silhouette_score(X[mask], y_db[mask])
            '''
            sil = np.nan

            mask = (y_db != -1)
            labels_kept = y_db[mask]

            n_samples = mask.sum()
            n_labels = len(np.unique(labels_kept))

            if n_samples >= 3 and 2 <= n_labels <= (n_samples - 1):
                sil = silhouette_score(X[mask], labels_kept)

            rows.append({
                "eps": eps,
                "min_samples": ms,
                "n_clusters": n_clusters,
                "noise_frac": noise_frac,
                "ARI": ari,
                "NMI": nmi,
                "silhouette_dbscan": sil,
            })
    return pd.DataFrame(rows)

def just_target(df):
    with open("trimmed_classes.pkl", "rb") as f:
        class_dict = pickle.load(f)
    labels = [class_dict[i] for i in range(len(class_dict))]
    labels_np = np.array(labels, dtype=object)
    scaler_t = StandardScaler()
    target_np = df.values.astype(np.float32)
    target_scaled = scaler_t.fit_transform(target_np)
    eps_list = np.logspace(-2, 1, 40)
    min_samples_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    df_target  = dbscan_sweep(target_scaled,  labels_np, eps_list, min_samples_list)
    df_target.to_csv(f"dbscan_eps_matrix_target_cond{sys.argv[2]}_2_8_3.csv", index=False)

def original_TSNE(df):
    tsne = TSNE(n_components=3, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    p = tsne.fit_transform(df)
    with open("trimmed_classes.pkl", "rb") as f:
        class_dict = pickle.load(f)
    labels = [class_dict[i] for i in range(len(class_dict))]
    #labels = [class_dict[i // 10] for i in range(len(p))] #shuffle labels to see if the scores 

    unique_classes = list(set(labels))
    colors = colormaps['tab20']
    class_to_color = {cls: colors(i) for i, cls in enumerate(unique_classes)}
    point_colors = [class_to_color[label] for label in labels]

    df_plot = pd.DataFrame(p, columns=["x", "y", "z"])
    df_plot["label"] = labels

    fig = px.scatter_3d(df_plot, x="x", y="y", z="z", color="label", title="TSNE of Latent Space", width=900, height=700, opacity=0.9) 
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

    fig.show()
    fig.write_html(f"tsne_MSE_10e_pred_TARGET.html") 


def main():
    df = pd.read_csv("predicted_trimmed.csv", header=None) # Per epoch change the noise
    target = torch.tensor(df.values, dtype=torch.float32) # 
    #just_target(df)
    #return
    #original_TSNE(df)
    #return
    dft = pd.read_csv("predicted_trimmed.csv", header=None)
    data = torch.tensor(dft.values, dtype=torch.float32)
    input_size = data.shape[1]
    final_size = target.shape[1]
    print(input_size)
    print(final_size)
    print(f"Input size: {input_size}")
    print(f"Final size: {final_size}")
    encoded_data = train_model_noise(data, target, input_size, final_size) #normalize latent space
    #encoded_data = []
    print(encoded_data.shape)
    print(encoded_data.T.shape)
    #print(net)
    #print(losses)
    #pca = PCA(n_components=3)
    tsne = TSNE(n_components=3, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    encoded_scaled = StandardScaler().fit_transform(encoded_data)
    encoded_norm = normalize(encoded_data)
    p = tsne.fit_transform(encoded_data)

    with open("trimmed_classes.pkl", "rb") as f:
        class_dict = pickle.load(f)
    labels = [class_dict[i] for i in range(len(class_dict))]
    #labels = [class_dict[i // 10] for i in range(len(p))] #shuffle labels to see if the scores 

    unique_classes = list(set(labels))
    colors = colormaps['tab20']
    class_to_color = {cls: colors(i) for i, cls in enumerate(unique_classes)}
    point_colors = [class_to_color[label] for label in labels]

    df_plot = pd.DataFrame(p, columns=["x", "y", "z"])
    df_plot["label"] = labels

    fig = px.scatter_3d(df_plot, x="x", y="y", z="z", color="label", title="TSNE of Latent Space", width=900, height=700, opacity=0.9) # try added ellipses around the same groups
    # Look into TSNE unscaled (1/20)

    fig.update_traces(marker=dict(size=6))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

    fig.show()
    fig.write_html(f"tsne_MSE_10e_pred_full_8_3.html") #hierarchal clustering
    # add noise in the columns
    '''
    df_coords = pd.DataFrame(p, columns=["tsne_x", "tsne_y", "tsne_z"])
    df_coords.insert(0, "point_num", np.arange(len(p)))
    df_coords["label"] = labels
    df_coords.to_csv("tsne_coords_8_3.csv", index=False)
    '''

    sil_score_data = silhouette_score(encoded_data, labels)
    sil_score_original = silhouette_score(target, labels)

    mod_sil_score = sil_score(encoded_data, labels)
    mod_sil_original = sil_score(target, labels)

    labels_np = np.array(labels, dtype=object)
    #kmeans_target = KMeans(n_clusters=6, random_state=42).fit(target) #try without manually setting the 6 clusters (harmonic clusters)
    #kmeans_encoded = KMeans(n_clusters=6, random_state=42).fit(encoded_data)

    scaler_t = StandardScaler() # Without scaling (1/20)
    scaler_e = StandardScaler()
    target_np = df.values.astype(np.float32)
    target_scaled = scaler_t.fit_transform(target_np)
    encoded_scaled = scaler_e.fit_transform(encoded_data)

    #eps_list = np.linspace(0.1, 3.0, 30)
    eps_list = np.logspace(-2, 1, 40)
    min_samples_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    df_target  = dbscan_sweep(target_scaled,  labels_np, eps_list, min_samples_list)
    df_encoded = dbscan_sweep(encoded_scaled, labels_np, eps_list, min_samples_list)

    df_target.to_csv(f"dbscan_eps_matrix_target_pred_full_8_3.csv", index=False)
    df_encoded.to_csv(f"dbscan_eps_matrix_encoded_pred_full_8_3.csv", index=False)

    '''
    dbscan = DBSCAN(eps=0.203, min_samples=3) #put in standard deviant #50 for min_sample (increments of 10)
    kmeans_target = dbscan.fit_predict(target_scaled)
    kmeans_encoded = dbscan.fit_predict(encoded_scaled)

    #Rand Index: Pairwise agreements (in the same class and cluster)
    rand_target = rand_score(labels_np, kmeans_target) #.labels_ for kmeans
    rand_encoded = rand_score(labels_np, kmeans_encoded)
    rand_together = rand_score(kmeans_target, kmeans_encoded)

    adj_rand_target = adjusted_rand_score(labels_np, kmeans_target)
    adj_rand_encoded = adjusted_rand_score(labels_np, kmeans_encoded)
    adj_rand_together = adjusted_rand_score(kmeans_target, kmeans_encoded)

    #Mutual Information: Amount of overlap between the clusterings
    mi_target = normalized_mutual_info_score(labels_np, kmeans_target)
    mi_encoded = normalized_mutual_info_score(labels_np, kmeans_encoded)
    mi_together = normalized_mutual_info_score(kmeans_target, kmeans_encoded)

    adj_mi_target = adjusted_mutual_info_score(labels_np, kmeans_target) #AMI corrects for chance
    adj_mi_encoded = adjusted_mutual_info_score(labels_np, kmeans_encoded)
    adj_mi_together = adjusted_mutual_info_score(kmeans_target, kmeans_encoded)

    with open("sil_MSE_10e_cond16_pred_750_300.txt", "w") as f:
      print("Sillhouette Score:\n", file=f)
      print("min_sample=20", file=f)
      print(f"Target: {sil_score_original}\tData: {sil_score_data}\n", file=f)
      print("Modified Silhouette:\n", file=f)
      print(f"Target: {mod_sil_original}\tData: {mod_sil_score}\n", file=f)
      print(f"Rand Index:\n", file=f)
      print(f"Target vs. Classes: {rand_target}\tData vs. Classes: {rand_encoded}\tTarget vs. Data: {rand_together}\n", file=f)
      print(f"Adjusted Rand Index:\n", file=f)
      print(f"Target vs. Classes: {adj_rand_target}\tData vs. Classes: {adj_rand_encoded}\tTarget vs. Data: {adj_rand_together}\n", file=f)
      print(f"Normalized Mutual Information\n", file=f)
      print(f"Target vs. Classes: {mi_target}\tData vs. Classes: {mi_encoded}\tTarget vs. Data: {mi_together}\n", file=f)
      print(f"Adjusted Mutual Information:\n", file=f)
      print(f"Target vs. Classes: {adj_mi_target}\tData vs. Classes: {adj_mi_encoded}\tTarget vs. Data: {adj_mi_together}\n", file=f)
      print("Number of classes:\n", file=f)
      print(f"Target: {len(np.unique(kmeans_target))}\tData: {len(np.unique(kmeans_encoded))}\n", file=f)
      print("--------------------\n", file=f)

 
    original_labels = labels_np
 
    for i in range(0, 3):
      random_labels = np.random.randint(0, len(np.unique(original_labels)), size=len(labels))
      print(np.unique(labels_np))
      labels_np = random_labels
      sil_score_data = silhouette_score(encoded_data, labels_np)
      sil_score_original = silhouette_score(target, labels_np)

      mod_sil_score = sil_score(encoded_data, labels_np)
      mod_sil_original = sil_score(target, labels_np)
   
      #kmeans_target = KMeans(n_clusters=6, random_state=42).fit(target)
      #kmeans_encoded = KMeans(n_clusters=6, random_state=42).fit(encoded_data)

      kmeans_target = dbscan.fit_predict(target)
      kmeans_encoded = dbscan.fit_predict(encoded_data)

      #Rand Index: Pairwise agreements (in the same class and cluster)
      rand_target = rand_score(labels_np, kmeans_target)
      rand_encoded = rand_score(labels_np, kmeans_encoded)
      rand_together = rand_score(kmeans_target, kmeans_encoded)

      adj_rand_target = adjusted_rand_score(labels_np, kmeans_target)
      adj_rand_encoded = adjusted_rand_score(labels_np, kmeans_encoded)
      adj_rand_together = adjusted_rand_score(kmeans_target, kmeans_encoded)

      #Mutual Information: Amount of overlap between the clusterings
      mi_target = normalized_mutual_info_score(labels_np, kmeans_target)
      mi_encoded = normalized_mutual_info_score(labels_np, kmeans_encoded)
      mi_together = normalized_mutual_info_score(kmeans_target, kmeans_encoded)

      adj_mi_target = adjusted_mutual_info_score(labels_np, kmeans_target) #AMI corrects for chance
      adj_mi_encoded = adjusted_mutual_info_score(labels_np, kmeans_encoded)
      adj_mi_together = adjusted_mutual_info_score(kmeans_target, kmeans_encoded)

      with open("sil_MSE_5e_2fn_std_200_32.txt", "a") as f:
        print("\n", file=f)
        print(f"RANDOM {i}\n", file=f)
        print("Sillhouette Score:\n", file=f)
        print(f"Target: {sil_score_original}\tData: {sil_score_data}\n", file=f)
        print("Modified Silhouette:\n", file=f)
        print(f"Target: {mod_sil_original}\tData: {mod_sil_score}\n", file=f)
        print(f"Rand Index:\n", file=f)
        print(f"Target vs. Classes: {rand_target}\tData vs. Classes: {rand_encoded}\tTarget vs. Data: {rand_together}\n", file=f)
        print(f"Adjusted Rand Index:\n", file=f)
        print(f"Target vs. Classes: {adj_rand_target}\tData vs. Classes: {adj_rand_encoded}\tTarget vs. Data: {adj_rand_together}\n", file=f)
        print(f"Normalized Mutual Information\n", file=f)
        print(f"Target vs. Classes: {mi_target}\tData vs. Classes: {mi_encoded}\tTarget vs. Data: {mi_together}\n", file=f)
        print(f"Adjusted Mutual Information:\n", file=f)
        print(f"Target vs. Classes: {adj_mi_target}\tData vs. Classes: {adj_mi_encoded}\tTarget vs. Data: {adj_mi_together}\n", file=f)

    for i in range(1, 11):
      scaler_t = StandardScaler()
      scaler_e = StandardScaler()
      target_scaled = scaler_t.fit_transform(target)
      encoded_scaled = scaler_e.fit_transform(encoded_data)

      dbscan = DBSCAN(eps=0.5, min_samples=1 + 2*i) #put in standard deviant #50 for min_sample (increments of 10)
      kmeans_target = dbscan.fit_predict(target_scaled)
      kmeans_encoded = dbscan.fit_predict(encoded_scaled)

      #Rand Index: Pairwise agreements (in the same class and cluster)
      rand_target = rand_score(labels_np, kmeans_target) #.labels_ for kmeans
      rand_encoded = rand_score(labels_np, kmeans_encoded)
      rand_together = rand_score(kmeans_target, kmeans_encoded)

      adj_rand_target = adjusted_rand_score(labels_np, kmeans_target)
      adj_rand_encoded = adjusted_rand_score(labels_np, kmeans_encoded)
      adj_rand_together = adjusted_rand_score(kmeans_target, kmeans_encoded)

      #Mutual Information: Amount of overlap between the clusterings
      mi_target = normalized_mutual_info_score(labels_np, kmeans_target)
      mi_encoded = normalized_mutual_info_score(labels_np, kmeans_encoded)
      mi_together = normalized_mutual_info_score(kmeans_target, kmeans_encoded)

      adj_mi_target = adjusted_mutual_info_score(labels_np, kmeans_target) #AMI corrects for chance
      adj_mi_encoded = adjusted_mutual_info_score(labels_np, kmeans_encoded)
      adj_mi_together = adjusted_mutual_info_score(kmeans_target, kmeans_encoded)
      
      with open("sil_MSE_10e_pred_200_32.txt", "a") as f:
        print(f"--MIN_SAMPLE={1 + 2*i}--", file=f)
        print("Sillhouette Score:\n", file=f)
        print(f"Target: {sil_score_original}\tData: {sil_score_data}\n", file=f)
        print("Modified Silhouette:\n", file=f)
        print(f"Target: {mod_sil_original}\tData: {mod_sil_score}\n", file=f)
        print(f"Rand Index:\n", file=f)
        print(f"Target vs. Classes: {rand_target}\tData vs. Classes: {rand_encoded}\tTarget vs. Data: {rand_together}\n", file=f)
        print(f"Adjusted Rand Index:\n", file=f)
        print(f"Target vs. Classes: {adj_rand_target}\tData vs. Classes: {adj_rand_encoded}\tTarget vs. Data: {adj_rand_together}\n", file=f)
        print(f"Normalized Mutual Information\n", file=f)
        print(f"Target vs. Classes: {mi_target}\tData vs. Classes: {mi_encoded}\tTarget vs. Data: {mi_together}\n", file=f)
        print(f"Adjusted Mutual Information:\n", file=f)
        print(f"Target vs. Classes: {adj_mi_target}\tData vs. Classes: {adj_mi_encoded}\tTarget vs. Data: {adj_mi_together}\n", file=f)
        print("Number of classes:\n", file=f)
        print(f"Target: {len(np.unique(kmeans_target))}\tData: {len(np.unique(kmeans_encoded))}\n", file=f)
        print("--------------------\n", file=f)



    
    print("Sillhouette Score:\n")
    print(f"Target: {sil_score_original}\tData: {sil_score_data}\n")
    print("Modified Silhouette:\n")
    print(f"Target: {mod_sil_original}\tData: {mod_sil_score}\n")
    print(f"Rand Index:\n")
    print(f"Target vs. Classes: {rand_target}\tData vs. Classes: {rand_encoded}\tTarget vs. Data: {rand_together}\n")
    print(f"Adjusted Rand Index:\n")
    print(f"Target vs. Classes: {adj_rand_target}\tData vs. Classes: {adj_rand_encoded}\tTarget vs. Data: {adj_rand_together}\n")
    print(f"Normalized Mutual Information\n")
    print(f"Target vs. Classes: {mi_target}\tData vs. Classes: {mi_encoded}\tTarget vs. Data: {mi_together}\n")
    print(f"Adjusted Mutual Information:\n")
    print(f"Target vs. Classes: {adj_mi_target}\tData vs. Classes: {adj_mi_encoded}\tTarget vs. Data: {adj_mi_together}\n")
 

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
