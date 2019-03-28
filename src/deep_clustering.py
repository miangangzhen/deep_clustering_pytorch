from train_utils import *


def main(doc_embeddings, n_cluster, pretrain=True, y=None, dims=None, device="cpu"):

    if dims is None:
        dims = [doc_embeddings.shape[-1], 500, 500, 2000, n_cluster]

    if pretrain:
        print("pretraining encoder and decoder to init encoder weights")
        pretrain_enc_dec(doc_embeddings, dims, n_epochs=300, batch_size=256, y_real=y, device=device)

    print("pretraining kmeans to init center points")
    init_center_with_kmeans(n_cluster, doc_embeddings, dims, y_real=y, device=device)

    print("run deep encoder clustering")
    return run_clustering(doc_embeddings, dims, n_epochs=300, y_real=y, batch_size=256, device=device)


if __name__ == "__main__":
    # GPU check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    labels = main(X, n_cluster, pretrain=is_pretrain, y=y, device=device)
