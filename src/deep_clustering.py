from sklearn.cluster import KMeans
from train_utils import *


def init_center_with_kmeans(n_clusters, X, dims):
    model = Encoder(dims)
    model.load_state_dict(torch.load("enc_dec_model"), strict=False)
    model.eval()
    with torch.no_grad():
        X_encoded = model(torch.from_numpy(X))

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
    kmeans.fit(X_encoded)
    np.save("cluster_centers.npy", kmeans.cluster_centers_)


def main(doc_embeddings, n_cluster, pretrain=True, y=None, dims=None):

    if dims is None:
        dims = [doc_embeddings.shape[-1], 500, 500, 2000, n_cluster]

    if pretrain:
        print("pretraining encoder and decoder to init encoder weights")
        pretrain_enc_dec(doc_embeddings, dims, n_epochs=300)
        print("pretraining kmeans to init center points")
        init_center_with_kmeans(n_cluster, doc_embeddings, dims)

    print("run deep encoder clustering")
    return run_clustering(doc_embeddings, dims, n_epochs=10, y_real=y)


if __name__ == "__main__":

    from torchvision import datasets
    train = datasets.MNIST("../data", train=True, download=True)
    test = datasets.MNIST("../data", train=False, download=True)
    (x_train, y_train), (x_test, y_test) = (train.train_data, train.train_labels), (test.test_data, test.test_labels)

    x = np.concatenate((x_train, x_test)).astype(np.float32)
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1)).astype(np.float32)
    X = np.divide(x, 255.)

    n_cluster = 10
    is_pretrain = True

    labels = main(X, n_cluster, pretrain=is_pretrain, y=y)
