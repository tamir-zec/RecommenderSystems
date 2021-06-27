import torch


class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, y, return_mean=True):
        mse = torch.pow(output - y, 2)

        if return_mean: return torch.mean(mse)
        return mse


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        eps = 1e-6
        criterion = torch.nn.MSELoss()
        loss = torch.sqrt(criterion(x, y) + eps)
        return loss
