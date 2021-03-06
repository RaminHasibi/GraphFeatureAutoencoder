import torch
from train_test import train_epoch, test
from models.End_to_End.nets import AE_MLP
def impute(model_class, data, opts):
    criterion = torch.nn.MSELoss()
    if model_class == AE_MLP:
        data.x = data.y = data.x.t()
        data.nonzeromask = data.nonzeromask.t()
    model = model_class(data.num_features, opts).to(opts.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
    for epoch in range(1, opts.epochs + 1):
        loss_train = train_epoch(model, data, optimizer, opts, criterion=criterion)
        if epoch%100 == 0:
            print('Exp: {:03d}, Loss: {:.5f}'.
                  format(epoch, loss_train))
    return model(data)
