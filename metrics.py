from torch.nn import BCEWithLogitsLoss
import torch
Nx = 341
Ny = 351

def Precision(y, y_pred):
    TP = torch.sum((y >= 0.5) * (y_pred > 0.5), axis=(1, 2)) * 1. / Nx / Ny
    FP = torch.sum((y >= 0.5) * (y_pred < 0.5), axis=(1, 2)) * 1. / Nx / Ny
    return torch.sum(TP / (TP + FP + .5e-4))


def Recall(y, y_pred):
    TP = torch.sum((y >= 0.5) * (y_pred > 0.5), axis=(1, 2)) * 1. / Nx / Ny
    FN = torch.sum((y < 0.5) * (y_pred > 0.5), axis=(1, 2)) * 1. / Nx / Ny
    return torch.sum(TP / (TP + FN + .5e-4))

def DiceScore(y_true, y_pred, gamma=1):
    nominator = 2 * \
                torch.sum(y_pred * y_true, axis=(1, 2)) + 1e-5
    denominator = torch.sum(y_pred ** gamma, axis=(1, 2)) \
                  + torch.sum(y_true ** gamma, axis=(1, 2)) + 1e-5
    result = nominator / denominator
    return torch.sum(result)


def DistCent(Y, Y_pred):
    x_p = torch.linspace(0, 1, Y.shape[-1])
    y_p = torch.linspace(0, 1, Y.shape[-2])

    x_true = torch.clone(Y.float())
    x_pred = torch.clone(Y_pred)
    for i in range(torch.numel(x_p)):
        x_true[:, :, i] *= x_p[i]
        x_pred[:, :, i] *= x_p[i]

    denom_true = torch.sum(Y, axis=(1, 2))
    denom_pred = torch.sum(Y_pred, axis=(1, 2))

    x_c_t = torch.sum(x_true, axis=(1, 2)) / denom_true
    x_c_p = torch.sum(x_pred, axis=(1, 2)) / denom_pred

    y_true = torch.clone(Y.float())
    y_pred = torch.clone(Y_pred)
    for i in range(torch.numel(y_p)):
        y_true[:, i, :] *= y_p[i]
        y_pred[:, i, :] *= y_p[i]

    y_c_t = torch.sum(y_true, axis=(1, 2)) / denom_true
    y_c_p = torch.sum(y_pred, axis=(1, 2)) / denom_pred

    return torch.sum(1 - (x_c_t - x_c_p) ** 2 - (y_c_t - y_c_p) ** 2)
