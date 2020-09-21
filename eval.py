import torch
from sklearn.metrics import mean_squared_error as scimse
import numpy as np
from sklearn.model_selection import KFold
from train_test import train_epoch, test
import copy
from models.End_to_End.nets import AE_MLP
from models.Embedding.model import Encoder
from utils.functions import index_to_mask


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
        train_feats_indeces, test_feats_indeces = next(kf_feats.split(np.arange(data.x.size(1))))
        eval_data.x = data.x[:, train_feats_indeces]
        eval_data.y = data.x[:, test_feats_indeces]
        eval_data.train_mask = index_to_mask(train_index, eval_data.x.size(0))
        eval_data.test_mask = index_to_mask(test_index, eval_data.x.size(0))
        for exp_num in range(eval_data.y.size(1)):
            torch.manual_seed(opts.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(opts.seed)
            model = model_class(eval_data.num_features, opts).to(opts.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
            for epoch in range(1, opts.epochs + 1):
                loss_train = train_epoch(model, eval_data, optimizer, opts, exp_num, criterion)
            loss_test = test(model, eval_data, exp_num, criterion, opts)
            model.eval()
            print('Exp: {:03d}, Loss: {:.10f}, TestLoss: {:.10f}'.
                  format(exp_num, loss_train, loss_test))
            with torch.no_grad():
                y_pred.append(model(eval_data))
#             del model
#             del optimizer
        for i in range(eval_data.y.size(1)):
            mse.append(scimse(y_pred[i][eval_data.test_mask.cpu().numpy()].cpu().numpy(),
                              eval_data.y[eval_data.test_mask, i].cpu().numpy().reshape([-1, 1])))
    print('Average+-std Error for test expression values: {:.10f}+-{:.10f}'.format(np.mean(mse), np.std(mse)))
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
        train_feats_indeces, test_feats_indeces = next(kf_feats.split(np.arange(data.x.size(1))))
        eval_data.x = data.x[:, train_feats_indeces]
        eval_data.y = data.x[:, test_feats_indeces]
        eval_data.train_mask = index_to_mask(train_index, eval_data.x.size(0))
        eval_data.test_mask = index_to_mask(test_index, eval_data.x.size(0))
        model = model_class(eval_data.num_features, eval_data.num_features).to(opts.device)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
        for epoch in range(1, opts.epochs + 1):
            train_epoch(model, eval_data, optimizer, opts)
        for exp_num in range(eval_data.y.size(1)):
            torch.manual_seed(opts.seed)
            z = model.encode(eval_data.x, eval_data.edge_index)
            model.fit_predictor(z[eval_data.train_mask].cpu().data.numpy(),
                                eval_data.y[eval_data.train_mask, exp_num].cpu().data.numpy())

            loss_test_lr, loss_test_rf = test(model, eval_data, exp_num, scimse, opts)
            model.eval()
            print('Exp: {:03d}, TestLoss_lr: {:.10f}, TestLoss_rf: {:.10f}'.
                  format(exp_num, loss_test_lr, loss_test_rf))
            with torch.no_grad():
                y_pred.append(model.predict(eval_data.x, eval_data.edge_index))
        #             del model
        #             del optimizer
        for i in range(eval_data.y.size(1)):
            mse_lr.append(scimse(y_pred[i][0][eval_data.test_mask.cpu().numpy()],
                              eval_data.y[eval_data.test_mask, i].cpu().numpy().reshape([-1, 1])))
            mse_rf.append(scimse(y_pred[i][1][eval_data.test_mask.cpu().numpy()],
                                 eval_data.y[eval_data.test_mask, i].cpu().numpy().reshape([-1, 1])))

    print('Average+-std Error for test expression values LR: {:.10f}+-{:.10f}'.format(np.mean(mse_lr), np.std(mse_lr)))
    print('Average+-std Error for test expression values RF: {:.10f}+-{:.10f}'.format(np.mean(mse_lr), np.std(mse_rf)))
    return mse_lr, mse_rf


def imputation_eval(model_class, data, opts):
    if model_class == AE_MLP:
        data.x = data.y = data.x.t()
        data.nonzeromask = data.nonzeromask.t()
    criterion = torch.nn.MSELoss()
    kf = KFold(n_splits=4, random_state=opts.seed, shuffle=True)
    loss_test = []
    if opts.dataset == 'Ecoli':
        indices = np.indices([data.x.size(0), data.x.size(1)]).reshape(2, -1)
    else:
        indices = np.array(data.x.data.numpy().nonzero())
    for k, train_test_indices in enumerate(kf.split(np.arange(len(indices[0])))):
        print('Fold number: {:d}'.format(k))
        train_index, test_index = train_test_indices
        eval_data = copy.deepcopy(data)
        eval_data.train_mask = index_to_mask([indices[0, train_index], indices[1, train_index]], eval_data.x.size())
        eval_data.test_mask = index_to_mask([indices[0, test_index], indices[1, test_index]], eval_data.x.size())
        model = model_class(eval_data.num_features, opts).to(opts.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
        for epoch in range(1, opts.epochs + 1):
            loss_train = train_epoch(model, eval_data, optimizer, opts, criterion=criterion)
        loss_test.appned(test(model, eval_data, None, criterion, opts))
        print('Loss: {:.5f}, TestLoss: {:.5f}'.
              format(loss_train, loss_test))
    return np.mean(loss_test)
