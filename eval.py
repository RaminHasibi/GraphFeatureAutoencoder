import torch
from sklearn.metrics import mean_squared_error as scimse
import numpy as np

from train import train_epoch, test


def prediction_eval(model_class, data, opts):
    y_pred = []
    loss_train = []
    criterion = torch.nn.MSELoss()
    for exp_num in range(data.y.size(1)):
        model = model_class(data.num_features, opts.out_channels, opts.hidden).to(opts.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
        for epoch in range(1, opts.epochs + 1):
            loss_train = train_epoch(model, data, optimizer, exp_num, criterion)
        loss_test = test(model, data, exp_num, criterion)
        model.eval()
        print('Exp: {:03d}, Loss: {:.5f}, TestLoss: {:.5f}'.
              format(exp_num, loss_train, loss_test))
        with torch.no_grad():
            y_pred.append(model(data))
        del model
        del optimizer
        mse = []
        for i in range(data.y.size(1)):
            mse.append(scimse(y_pred[i][data.test_mask.cpu().data.numpy()].cpu().data.numpy(),
                              data.y[data.test_mask, i].cpu().data.numpy().reshape([-1, 1])))
        print('Average+-std Error for test expression values: {:.5f}+-{:.5f}'.format(np.mean(mse), np.std(mse)))
