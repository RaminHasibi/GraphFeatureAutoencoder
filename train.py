import torch


def train_epoch(model, data, optimizer, exp_num, criterion):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output[data.train_mask], data.y[data.train_mask, exp_num].reshape([-1, 1]))
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, exp_num, criterion):
    model.eval()
    output = model(data)
    loss = criterion(output[data.test_mask], data.y[data.test_mask, exp_num].reshape([-1, 1]))
    return loss.item()