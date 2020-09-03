import torch
from sklearn.metrics import mean_squared_error as scimse
import numpy as np
from sklearn.model_selection import KFold
from train_test import train_epoch, test
import copy

from utils.functions import index_to_mask

def prediction_eval(model_class, data, opts):

    loss_train = []
    criterion = torch.nn.MSELoss()
    kf = KFold(n_splits=3)
    kf_feats = KFold(n_splits=3)

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
            model = model_class(eval_data.num_features, opts).to(opts.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
            for epoch in range(1, opts.epochs + 1):
                loss_train = train_epoch(model, eval_data, optimizer, exp_num, criterion ,opts)
            loss_test = test(model, eval_data, exp_num, criterion, opts)
            model.eval()
            print('Exp: {:03d}, Loss: {:.5f}, TestLoss: {:.5f}'.
                  format(exp_num, loss_train, loss_test))
            with torch.no_grad():
                y_pred.append(model(eval_data))
            del model
            del optimizer
        for i in range(eval_data.y.size(1)):
            mse.append(scimse(y_pred[i][eval_data.test_mask.cpu().eval_data.numpy()].cpu().eval_data.numpy(),
                              eval_data.y[eval_data.test_mask, i].cpu().eval_data.numpy().reshape([-1, 1])))
    print('Average+-std Error for test expression values: {:.5f}+-{:.5f}'.format(np.mean(mse), np.std(mse)))
    return mse


def imputation_eval(model_class, data, opts):
    criterion = torch.nn.MSELoss()
    kf = KFold(n_splits=4)
    loss_test = []
    if opts.dataset == 'Ecoli':
        indices = np.indices([data.x.size(0), data.x.size(1)]).reshape(2, -1)
    else:
        matrix_mask = torch.zeros([data.x.size(0), data.x.size(1)])
        indices = np.array(data.x.data.numpy().nonzero())
        matrix_mask[data.x.nonzero(as_tuple=True)] = 1
    for k, train_test_indices in enumerate(kf.split(np.arange(len(indices[0])))):
        print('Fold number: {:d}'.format(k))
        train_index, test_index = train_test_indices
        eval_data = copy.deepcopy(data)
        matrix_mask = torch.zeros([eval_data.x.size(0), eval_data.x.size(1)])


        eval_data.train_mask = index_to_mask([indices[0, train_index], indices[1, train_index]], eval_data.x.size())
        eval_data.test_mask = index_to_mask([indices[0, test_index], indices[1, test_index]], eval_data.x.size())
        model = model_class(eval_data.num_features, opts).to(opts.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
        for epoch in range(1, opts.epochs + 1):
            loss_train = train_epoch(model, eval_data, optimizer, None, criterion, opts)
        loss_test.appned(test(model, eval_data, None, criterion, opts))
        print('Loss: {:.5f}, TestLoss: {:.5f}'.
              format(loss_train, loss_test))
    return np.mean(loss_test)
