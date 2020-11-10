import torch
from sklearn.metrics import mean_squared_error as scimse
from torch_geometric.utils import to_undirected
import numpy as np
import networkx as nx
from sklearn.model_selection import KFold
from train_test import train_epoch, test
import copy
from models.End_to_End.nets import AE_MLP
from models.Embedding.model import Encoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from utils.functions import index_to_mask
from magic import MAGIC

def supervised_prediction_eval(model_class, data, opts):

    loss_train = []
    criterion = torch.nn.MSELoss()
    kf = KFold(n_splits=3, random_state=opts.seed)
    kf_feats = KFold(n_splits=3, random_state=opts.seed)

    mse = []

    for k, train_test_indices in enumerate(kf.split(data.x)):
        print('Fold number: {:d}'.format(k))
        y_pred = []
        train_index, test_index = train_test_indices
        eval_data = copy.deepcopy(data)
        if opts.random_graph:
            print('Random Graph used')
            G_rand = nx.gnp_random_graph(data.x.shape[0],opts.random_graph_alpha)
            eval_data.edge_index = to_undirected(torch.tensor(np.array(G_rand.edges()).T).to(opts.device))
            print(eval_data)
        train_feats_indeces, test_feats_indeces = next(kf_feats.split(np.arange(data.y.size(1))))
        if not opts.no_features:
            eval_data.x = data.x[:, train_feats_indeces]
        eval_data.y = data.y[:, test_feats_indeces]
        eval_data.train_mask = index_to_mask(train_index, eval_data.x.size(0))
        eval_data.test_mask = index_to_mask(test_index, eval_data.x.size(0))
        for exp_num in range(eval_data.y.size(1)):
            if (model_class == LinearRegression) | (model_class == RandomForestRegressor):
                model = model_class()
                model.fit(eval_data.x[eval_data.train_mask], eval_data.y[eval_data.train_mask, exp_num])
                pred = model.predict(eval_data.x[eval_data.test_mask])
                test_loss = scimse(pred,
                       eval_data.y[eval_data.test_mask, exp_num])
                print('Exp: {:03d}, Loss: {:.5f}'
                      .format(exp_num, test_loss))
                y_pred.append(pred)
            else:
                torch.manual_seed(opts.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(opts.seed)

                model = model_class(eval_data.num_features, opts).to(opts.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
                best_loss = 1e9
                for epoch in range(1, opts.epochs + 1):
                    loss_train = train_epoch(model, eval_data, optimizer, opts, exp_num, criterion)
                    if loss_train < best_loss:
                        best_loss = loss_train
                        best_model = copy.deepcopy(model)
                loss_test = test(best_model, eval_data, exp_num, criterion, opts)
                print('Exp: {:03d}, Loss: {:.5f}, TestLoss: {:.5f}'.
                      format(exp_num, loss_train, loss_test))
                with torch.no_grad():
                    y_pred.append(best_model(eval_data))

        for i in range(eval_data.y.size(1)):
            if (model_class == LinearRegression) | (model_class == RandomForestRegressor):
                mse.append(scimse(y_pred[i],
                                  eval_data.y[eval_data.test_mask, i]))
            else:
                mse.append(scimse(y_pred[i][eval_data.test_mask.cpu().numpy()].cpu().numpy(),
                                  eval_data.y[eval_data.test_mask, i].cpu().numpy().reshape([-1, 1])))
    print('Average+-std Error for test expression values: {:.5f}+-{:.5f}'.format(np.mean(mse), np.std(mse)))
    return mse

def embedding_prediction_eval(model_class, data, opts):
    loss_train = []

    kf = KFold(n_splits=3, random_state=opts.seed, shuffle=True)
    kf_feats = KFold(n_splits=3, random_state=opts.seed, shuffle=True)

    mse_lr = []
    mse_rf = []

    for k, train_test_indices in enumerate(kf.split(data.x)):
        print('Fold number: {:d}'.format(k))
        y_pred = []
        train_index, test_index = train_test_indices
        eval_data = copy.deepcopy(data)
        train_feats_indeces, test_feats_indeces = next(kf_feats.split(np.arange(data.y.size(1))))
        if not opts.no_features:
            eval_data.x = data.x[:, train_feats_indeces]
        eval_data.y = data.y[:, test_feats_indeces]
        eval_data.train_mask = index_to_mask(train_index, eval_data.x.size(0))
        eval_data.test_mask = index_to_mask(test_index, eval_data.x.size(0))
        model = model_class(eval_data.num_features, 32).to(opts.device)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
        print('Training the auto encoder!')
        for epoch in range(1, opts.epochs + 1):
            if epoch % 10 == 0:
                print('Epoch number: {:03d}'.format(epoch))
            train_epoch(model, eval_data, optimizer, opts)
        for exp_num in range(eval_data.y.size(1)):
            torch.manual_seed(opts.seed)
            z = model.encode(eval_data.x, eval_data.edge_index)
            model.fit_predictor(z[eval_data.train_mask].cpu().data.numpy(),
                                eval_data.y[eval_data.train_mask, exp_num].cpu().data.numpy())

            loss_test_lr, loss_test_rf = test(model, eval_data, exp_num, scimse, opts)
            model.eval()
            print('Exp: {:03d}, TestLoss_lr: {:.5f}, TestLoss_rf: {:.5f}'.
                  format(exp_num, loss_test_lr, loss_test_rf))
            with torch.no_grad():
                y_pred.append(model.predict(eval_data.x, eval_data.edge_index))
        for i in range(eval_data.y.size(1)):
            mse_lr.append(scimse(y_pred[i][0][eval_data.test_mask.cpu().numpy()],
                              eval_data.y[eval_data.test_mask, i].cpu().numpy().reshape([-1, 1])))
            mse_rf.append(scimse(y_pred[i][1][eval_data.test_mask.cpu().numpy()],
                                 eval_data.y[eval_data.test_mask, i].cpu().numpy().reshape([-1, 1])))

    print('Average+-std Error for test expression values LR: {:.5f}+-{:.5f}'.format(np.mean(mse_lr), np.std(mse_lr)))
    print('Average+-std Error for test expression values RF: {:.5f}+-{:.5f}'.format(np.mean(mse_rf), np.std(mse_rf)))
    return mse_lr, mse_rf


def imputation_eval(model_class, data, opts):
    if model_class == MAGIC:
        data.x = data.y = data.x.t()
        data.nonzeromask = data.nonzeromask.t()
    criterion = torch.nn.MSELoss()
    kf = KFold(n_splits=3, random_state=opts.seed, shuffle=True)
    loss_test = []
    if opts.dataset == 'Ecoli':
        indices = np.indices([data.x.size(0), data.x.size(1)]).reshape(2, -1)
    else:
        indices = np.array(data.x.cpu().data.numpy().nonzero())
    for k, train_test_indices in enumerate(kf.split(np.arange(len(indices[0])))):
        print('Fold number: {:d}'.format(k))
        train_index, test_index = train_test_indices
        eval_data = copy.deepcopy(data)
        eval_data.train_mask = index_to_mask([indices[0, train_index], indices[1, train_index]],
                                             eval_data.x.size()).to(opts.device)
        eval_data.test_mask = index_to_mask([indices[0, test_index], indices[1, test_index]],j
                                            eval_data.x.size()).to(opts.device)
        if model_class == MAGIC:
            pred = model_class().fit_transform((eval_data.x*eval_data.train_mask).cpu().data.numpy())
            loss_test.append(scimse(pred*eval_data.test_mask.cpu().data.numpy(),
                                    (eval_data.y*eval_data.test_mask).cpu().data.numpy()))
        else:
            model = model_class(eval_data.num_features, opts).to(opts.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
            for epoch in range(1, opts.epochs + 1):
                loss_train = train_epoch(model, eval_data, optimizer, opts, criterion=criterion)
                if epoch % 10 == 0:
                    print('Epoch number: {:03d}, Train_loss: {:.5f}'.format(epoch, loss_train))
            loss_test.append(test(model, eval_data, None, criterion, opts))
            print('Loss: {:.5f}, TestLoss: {:.5f}'.
                    format(loss_train, loss_test[k]))
    print('Average+-std Error for test RNA values: {:.5f}+-{:.5f}'.format(np.mean(loss_test), np.std(loss_test)))
    return np.mean(loss_test)
