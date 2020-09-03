import torch


def train_epoch(model, data, optimizer, exp_num, criterion ,opts):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    if opts.problem == 'Prediction':
        loss = criterion(output[data.train_mask], data.y[data.train_mask, exp_num].reshape([-1, 1]))
    else:
        loss = criterion(output * (data.nonzeromask), data.y * (data.nonzeromask))
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, exp_num, criterion, opts):
    model.eval()
    output = model(data)
    if opts.problem == 'Prediction':
        loss = criterion(output[data.test_mask], data.y[data.test_mask, exp_num].reshape([-1, 1]))
    else:
        loss = criterion(output*data.test_mask, data.y*data.test_mask)
    return loss.item()
