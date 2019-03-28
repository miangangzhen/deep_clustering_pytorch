import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, dims):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i != len(dims) - 2:
                self.encoder_layers.append(nn.LeakyReLU())
                # self.encoder_layers.append(nn.Dropout(p=0.1))

    def forward(self, x):
        encode = x
        for layer in self.encoder_layers:
            encode = layer(encode)
        return encode


class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            self.decoder_layers.append(nn.Linear(dims[i], dims[i-1]))
            if i != 1:
                self.decoder_layers.append(nn.LeakyReLU())
                # self.decoder_layers.append(nn.Dropout(p=0.1))

    def forward(self, x):
        decode = x
        for layer in self.decoder_layers:
            decode = layer(decode)
        return decode


class Clustering(nn.Module):
    """
        Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
        sample belonging to each cluster. The probability is calculated with student's t-distribution.

        # Example
        ```
            model.add(ClusteringLayer(n_clusters=10))
        ```
        # Arguments
            n_clusters: number of clusters.
            weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
            alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
        # Input shape
            2D tensor with shape: `(n_samples, n_features)`.
        # Output shape
            2D tensor with shape: `(n_samples, n_clusters)`.
        """
    def __init__(self, alpha=1.0):
        super(Clustering, self).__init__()
        arr = np.load("cluster_centers.npy").astype(np.float32)
        self.clusters = nn.Parameter(torch.from_numpy(arr))
        self.alpha = alpha

    def forward(self, x):
        """ student t-distribution, as same as used in t-SNE algorithm.
             Measure the similarity between embedded point z_i and centroid µ_j.
                     q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                     q_ij can be interpreted as the probability of assigning sample i to cluster j.
                     (i.e., a soft assignment)
            Arguments:
                inputs: the variable containing data, shape=(n_samples, n_features)
            Return:
                q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
            """
        # x = [batch_size, hidden_size] => [batch_size, 1, hidden_size]
        # => [batch_size, n_clusters, hidden_size]
        # => q = [batch_size, hidden_size]
        q = 1.0 / (1.0 + (torch.sum(torch.pow(torch.unsqueeze(x, 1) - self.clusters, 2), 2) / self.alpha))
        q = q.pow((self.alpha + 1.0) / 2.0)
        # 注意，这里没用softmax，因为softmax会对值进行exp非线性变换。而这里是线性变换。
        q = torch.transpose(torch.transpose(q, 1, 0) / torch.sum(q, 1), 1, 0)  # Make sure each sample's k values add up to 1.
        return q


class HybridModel(nn.Module):
    def __init__(self, dims):
        super(HybridModel, self).__init__()
        self.encoder = Encoder(dims)
        self.clustering = Clustering()
        self.decoder = Decoder(dims)

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        clu = self.clustering(enc)
        return dec, clu
