
import torch
from torch import nn
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
# from sacred import Ingredient
import numpy as np

from algorithms.mappo.algorithms.r_mappo.algorithm.vae import LinearVAE
import wandb
import math

# Initialize a new run
# wandb.init(project="vae_classification", entity="2017920898")


def normalize_data(buffer):
    # Compute mean and standard deviation
    mean = buffer.mean(axis=0)
    std = buffer.std(axis=0)

    # Avoid division by zero by setting std to 1 where it's 0
    # std[std == 0] = 1

    # Normalize the buffer
    normalized_buffer = (buffer - mean) / (std+1e-6)
    return normalized_buffer
def compute_clusters(rb, agent_count, batch_size, clusters, lr, epochs, z_features, kl_weight, device):
    # print("here!!!!",f"{lr:.5e}")
    # device = 'cpu'
    device =device
    # dataset = rbDataSet(rb,encoder_in,decoder_in,reconstruct)
    # print("rb.obs_buffer.shape",type(rb.obs_buffer))
    # print("rb.one_hot_list_buffer.shape",rb.one_hot_list_buffer.shape)
    # print("rb.actions_buffer.shape",rb.actions_buffer.shape)
    # print("rewards_buffer.shape",rb.rewards_buffer.shape)
    obs_buffer_flat = rb.obs_buffer.reshape(-1, rb.obs_buffer.shape[-1])  # Flatten the first two dimensions
    actions_buffer_flat = rb.actions_buffer.reshape(-1, rb.actions_buffer.shape[-1])
    rewards_buffer_flat = rb.rewards_buffer.reshape(-1, rb.rewards_buffer.shape[-1])
    agent_flat = rb.one_hot_list_buffer.reshape(-1, rb.one_hot_list_buffer.shape[-1])
    next_obs_buffer_flat = rb.next_obs_buffer.reshape(-1, rb.next_obs_buffer.shape[-1])
    # print(next_obs_buffer_flat)
    actions_buffer_flat = normalize_data(actions_buffer_flat)
    obs_buffer_flat = normalize_data(obs_buffer_flat)
    rewards_buffer_flat = normalize_data(rewards_buffer_flat)
    next_obs_buffer_flat = normalize_data(next_obs_buffer_flat)
    # print(obs_buffer_flat.shape, actions_buffer_flat.shape, rewards_buffer_flat.shape, agent_flat.shape)
    extra_decoder = np.concatenate([obs_buffer_flat, actions_buffer_flat], axis=-1)
    reconstruct = np.concatenate([next_obs_buffer_flat, rewards_buffer_flat], axis=-1)
    # print(extra_decoder.shape, reconstruct.shape)
    extra_decoder_tensor = torch.tensor(extra_decoder, dtype=torch.float32).to(device)

    reconstruct_tensor = torch.tensor(reconstruct, dtype=torch.float32).to(device)
    encode_tensor= torch.tensor(agent_flat, dtype=torch.float32).to(device)
    # print(encode_tensor.shape,extra_decoder_tensor.shape,reconstruct_tensor.shape)


    input_size = rb.one_hot_list_buffer.shape[-1]
    extra_decoder_input = rb.obs_buffer.shape[-1]+rb.actions_buffer.shape[-1]
    reconstruct_size = rb.next_obs_buffer.shape[-1]+rb.rewards_buffer.shape[-1]
    # print(input_size,extra_decoder_input,reconstruct_size)
    # input_size = dataset.data[0].shape[-1]
    # extra_decoder_input = dataset.data[1].shape[-1]
    # reconstruct_size = dataset.data[2].shape[-1]
    
    
    model = LinearVAE(z_features, input_size, extra_decoder_input, reconstruct_size)
    print(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # criterion = nn.BCELoss(reduction='sum')
    criterion = nn.MSELoss(reduction="sum")
    def final_loss(bce_loss, mu, logvar):
        """
        This function will add the reconstruction loss (BCELoss) and the 
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """
        # see equation 1
        BCE = bce_loss 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # 
        return BCE + kl_weight*KLD

    def fit(model, encode_tensor, extra_decoder_tensor, reconstruct_tensor, batch_size):
        model.train()
        running_loss = 0.0
        num_samples = encode_tensor.size(0)
        # print(num_samples)
        for i in range(0, num_samples, batch_size):
            # Extracting the batch
            batch_encode = encode_tensor[i:i+batch_size]
            batch_extra_decoder = extra_decoder_tensor[i:i+batch_size]
            batch_reconstruct = reconstruct_tensor[i:i+batch_size]

            optimizer.zero_grad()
            reconstruction, mu, logvar = model(batch_encode, batch_extra_decoder)
            bce_loss = criterion(reconstruction, batch_reconstruct)
            loss = final_loss(bce_loss, mu, logvar)
            wandb.log({"loss": loss/num_samples})
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            # print(loss)
        
        return running_loss / num_samples
    train_loss = []
    # Training loop
    for epoch in tqdm(range(epochs)):
        train_epoch_loss = fit(model, encode_tensor, extra_decoder_tensor, reconstruct_tensor, batch_size)
        train_loss.append(train_epoch_loss)

    # check? what is agent count?
    x = torch.eye(agent_count).to(device)
    # This step is crucial as it provides 
    # the latent representations corresponding to each agent

    with torch.no_grad():
        z = model.encode(x)
    z_np = z.to('cpu').numpy()
    z_np = z_np[:, :]
    # what is cluster here?

    if clusters is None:
        clusters = find_optimal_cluster_number(z_np)
    print(clusters)
    # _log.info(f"Creating {clusters} clusters.")
    # run k-means from scikit-learn
    kmeans = KMeans(
        n_clusters=clusters, init='k-means++',
        n_init=10
    )
    cluster_ids_x = kmeans.fit_predict(z_np) # predict labels
    # if z_features == 2:
    #     plot_clusters(kmeans.cluster_centers_, z)
    return torch.from_numpy(cluster_ids_x).long()




def find_optimal_cluster_number(X):

    range_n_clusters = list(range(2, X.shape[0]))
    scores = {}

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        scores[n_clusters] = davies_bouldin_score(X, cluster_labels)

    max_key = min(scores, key=lambda k: scores[k])
    return max_key