from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import metrics
from models import *
import os


def pretrain_enc_dec(doc_embeddings, dims, batch_size=16, n_epochs=10):

    inputs = torch.from_numpy(doc_embeddings)
    dataset = TensorDataset(inputs, inputs)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    model = nn.Sequential(Encoder(dims), Decoder(dims))
    # print(model)
    # for param in model.parameters():
    #     print(type(param.data), param.size())

    if os.path.exists("enc_dec_model"):
        model.load_state_dict(torch.load("enc_dec_model"))
        print("encoder-decoder model load from ckpt")

    model.train()

    optimizer = Adam(model.parameters(), lr=1e-4)

    criterion = nn.SmoothL1Loss()

    bst_model_loss = 999.9
    for epoch in range(n_epochs):
        train_loss = 0.0
        for i, data in enumerate(dataloader):
            x, y = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if (i+1) % 100 == 0:
                mean_loss = train_loss / 100
                print("epoch: {}, step loss: {}".format(epoch+1, mean_loss))
                if mean_loss < bst_model_loss:
                    torch.save(model.state_dict(), "enc_dec_model")
                    bst_model_loss = mean_loss
                train_loss = 0.0


# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


def run_clustering(doc_embeddings, dims, batch_size=16, n_epochs=1, update_interval=500, tol=0.001, y_real=None):

    inputs = torch.from_numpy(doc_embeddings)
    dataset = TensorDataset(inputs)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    model = HybridModel(dims)
    enc_dec_model = {k[2:]: v for k, v in torch.load("enc_dec_model").items()}
    model.encoder.load_state_dict(enc_dec_model, strict=False)
    model.decoder.load_state_dict(enc_dec_model, strict=False)

    if os.path.exists("clustering_model"):
        model.load_state_dict(torch.load("clustering_model"))
        print("clustering model load from ckpt")

    model.train()

    optimizer = Adam(model.parameters(), lr=1e-3)

    criterion1 = nn.KLDivLoss(reduction="batchmean")
    criterion2 = nn.SmoothL1Loss()

    y_pred_last = np.zeros([doc_embeddings.shape[0]])

    is_end = False
    bst_model_acc = 0.0
    for epoch in range(n_epochs):
        if is_end:
            break
        batch_num = 1
        train_loss = 0.0
        for data in dataloader:

            if (batch_num-1) % update_interval == 0:
                model.eval()
                with torch.no_grad():
                    _, q = model(inputs)
                    p = target_distribution(q.numpy())
                y_pred = q.numpy().argmax(1)

                if y_real is not None:
                    acc = np.round(metrics.acc(y_real, y_pred), 5)
                    nmi = np.round(metrics.nmi(y_real, y_pred), 5)
                    ari = np.round(metrics.ari(y_real, y_pred), 5)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % ((epoch+1)*batch_num, acc, nmi, ari))
                    if acc > bst_model_acc:
                        torch.save(model.state_dict(), "clustering_model")
                        bst_model_acc = acc

                # check stop criterion - model convergence
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                print("delta_label: {}".format(delta_label))
                y_pred_last = np.copy(y_pred)
                model.train()

                if delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    is_end = True
                    break

            x_batch = data[0]
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            y_hat_dec_batch, y_hat_clu_batch = model(x_batch)
            y_batch = p[((batch_num - 1) * batch_size):(batch_num * batch_size), :]
            loss1 = 1e-1 * criterion1(torch.log(y_hat_clu_batch), torch.from_numpy(y_batch))
            loss2 = criterion2(y_hat_dec_batch, x_batch)
            loss = loss1 + loss2
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_num % 100 == 0:
                print("kld loss: {}, mse loss: {}".format(loss1, loss2))
                print("step loss: {}".format(train_loss / 100))
                train_loss = 0.0
            batch_num += 1

    torch.save(model.state_dict(), "clustering_model")

    model.eval()
    with torch.no_grad():
        q = model(inputs).numpy()
    return q.argmax(1)
