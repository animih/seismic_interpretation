import torch
from IPython.display import clear_output
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Loss_Metrics_Mem(object):

    def __init__(self):
        super().__init__()
        self.loss = None
        self.loss_h = []
        self.val_loss_h = []
        self.metrics_h = {}

    def upd_loss(self, loss_val):
        if self.loss is None:
            self.loss = loss_val
        else:
            self.loss = self.loss + loss_val

    def clean_loss(self):
        self.loss = None

    def upd_metrics(self, loss_val, metrics_val: dict):
        self.val_loss_h.append(loss_val)

        for key in metrics_val.keys():
            if key in self.metrics_h.keys():
                self.metrics_h[key].append(metrics_val[key])
            else:
                self.metrics_h[key] = [metrics_val[key]]

        self.loss_h.append(self.loss)

    def plot(self, axes):
        xlen = len(self.loss_h)
        axes[0].plot(range(xlen), self.loss_h, label='train')
        axes[0].plot(range(xlen), self.val_loss_h, label='test')
        axes[0].legend()

        for key in self.metrics_h.keys():
            axes[1].plot(range(xlen), self.metrics_h[key], label=key)

        axes[1].legend()

def train_detection(model, optimizer, train_dataloader, test_dataloader,
                    train_len, test_len,
                    Loss, metrics_dict = {}, N_epochs = 150):

    l_m = Loss_Metrics_Mem()
    
    for epoch in range(N_epochs):
        loss_val = 0
        model.train()
        for X, y, _ in train_dataloader:
            X = X.float()
            y = y.squeeze()
            y_pred = model(X.to(device)).squeeze()
            loss = Loss(y_pred.float(), y.to(device).float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            l_m.upd_loss(loss.item() * y.shape[0] / train_len)

        metric_val = {}
        for key in metrics_dict.keys():
            metric_val[key] = 0
        #model.eval()
        with torch.no_grad():
            for X, y, _ in test_dataloader:
                X = X.float()
                y = y.squeeze()
                y_pred = model(X.to(device)).squeeze()
                loss = Loss(y_pred.float(), y.to(device).float())
                loss_val += loss.item() * y.shape[0] / test_len

                for key in metrics_dict.keys():
                    metric_val[key] += metrics_dict[key](y, torch.sigmoid(y_pred).to('cpu').detach()).item() / test_len

        l_m.upd_metrics(loss_val,metric_val)

        clear_output()
        # plot losses
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        axes[0].set_xlabel('N Epochs')
        axes[0].grid()
        axes[1].grid()
        axes[0].set_title('Loss')
        axes[0].set_yscale('log')
        axes[1].set_xlabel('N Epochs')
        axes[1].set_title('Metrics')
        l_m.plot(axes)
        plt.show()

        print(f'Loss on validation : {l_m.val_loss_h[-1]} \n')
        for key, val in metric_val.items():
            print(f'{key} : {val}  \n')

        l_m.clean_loss()

    return model, l_m

def train_regression(model, optimizer, train_dataloader, test_dataloader,
                     train_len, test_len,
                    Loss, metrics_dict = {}, N_epochs = 85):
    l_m = Loss_Metrics_Mem()

    for epoch in range(N_epochs):

        y_pred_val = torch.tensor([0])
        y_true_val = torch.tensor([0])

        loss_val = 0

        model.train()
        for X, _, y in train_dataloader:
            X = X.float()
            X_out = model(X.to(device)).squeeze()
            loss = Loss(X_out.float(), y.to(device).float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            l_m.upd_loss(loss.item() * torch.numel(y) / train_len)
        metric_val = {}
        for key in metrics_dict.keys():
            metric_val[key] = 0
        # model.eval()
        with torch.no_grad():
            for X, _, y in test_dataloader:
                X = X.float()
                y_pred = model(X.to(device)).squeeze().to('cpu')
                loss = Loss(y.float(), y_pred.float())

                loss_val += loss.item() * torch.numel(y) / test_len

                if torch.numel(y_pred) > 1:
                    y_pred_val = torch.cat((y_pred_val, y_pred), dim=0)
                    y_true_val = torch.cat((y_true_val, y.to('cpu')), dim=0)
        for key in metrics_dict.keys():
            metric_val[key] += metrics_dict[key](y_pred_val.squeeze().detach().cpu().numpy(),
                                 y_true_val.detach().cpu().numpy())

        l_m.upd_metrics(loss_val,metric_val)

        clear_output()
        # plot losses
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        axes[0].set_xlabel('N Epochs')
        axes[0].grid()
        axes[1].grid()
        axes[0].set_title('Loss')
        axes[0].set_yscale('log')
        axes[1].set_xlabel('N Epochs')
        axes[1].set_title('Metrics')
        l_m.plot(axes)
        plt.show()

        print(f'Loss on validation : {l_m.val_loss_h[-1]} \n')
        for key, val in metric_val.items():
            print(f'{key} : {val}  \n')

        l_m.clean_loss()

    return model, l_m

def train_multitask(model, optimizer, train_dataloader, test_dataloader,
                    train_len, test_len,
                    Loss1, Loss2, metrics_dict1 = {}, metrics_dict2 = {}, N_epochs = 150):
    l_m1 = Loss_Metrics_Mem()
    l_m2 = Loss_Metrics_Mem()

    for epoch in range(N_epochs):

        y_pred_val = torch.tensor([0])
        y_true_val = torch.tensor([0])

        loss_val = 0
        loss_val_s = 0

        model.train()
        for X, y, y_s in train_dataloader:
            X = X.float()
            y1, y2 = model(X.to(device))
            y1 = y1.squeeze()
            y2 = y2.squeeze()
            loss1 = Loss1(y1.float(), y.to(device).float())
            loss2 = Loss2(y2.float(), y_s.to(device).float())

            loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            l_m1.upd_loss(loss1.item() * y.shape[0] / train_len)
            l_m2.upd_loss(loss2.item() * torch.numel(y_s) / train_len)
        # model.eval()
        metric_val1 = {}
        for key in metrics_dict1.keys():
            metric_val1[key] = 0.
        metric_val2 = {}
        for key in metrics_dict2.keys():
            metric_val2[key] = 0.
        with torch.no_grad():
            for X, _, y in test_dataloader:
                X = X.float()
                _, y_pred = model(X.to(device))
                y_pred = y_pred.squeeze().to('cpu')
                loss = Loss2(y.float(), y_pred.float())

                loss_val_s += loss.item() * torch.numel(y) / test_len

                if torch.numel(y_pred) > 1:
                    y_pred_val = torch.cat((y_pred_val, y_pred), dim=0)
                    y_true_val = torch.cat((y_true_val, y.to('cpu')), dim=0)
        with torch.no_grad():
            for X, y, _ in test_dataloader:
                X = X.float()
                y = y.squeeze()
                y_pred, _ = model(X.to(device))
                y_pred = y_pred.squeeze().to('cpu')
                loss = Loss1(y_pred.to(device).float(), y.to(device).float())
                loss_val += loss.item() * y.shape[0] / test_len
                for key in metrics_dict1.keys():
                    metric_val1[key] += metrics_dict1[key](y, torch.sigmoid(y_pred).to('cpu').detach()).item() / test_len

        for key in metrics_dict2.keys():
            metric_val2[key] += metrics_dict2[key](y_pred_val.squeeze().detach().cpu().numpy(),
                                 y_true_val.detach().cpu().numpy())

        l_m1.upd_metrics(loss_val,metric_val1)

        l_m2.upd_metrics(loss_val_s,metric_val2)

        clear_output()
        # plot losses
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        axes[0].set_xlabel('N Epochs')
        axes[0].grid()
        axes[1].grid()
        axes[0].set_title('Loss')
        axes[0].set_yscale('log')
        axes[1].set_xlabel('N Epochs')
        axes[1].set_title('Metrics')
        l_m1.plot(axes)
        plt.show()

        print(f'Loss on validation : {l_m1.val_loss_h[-1]} \n')
        for key, val in metric_val1.items():
            print(f'{key} : {val}  \n')

        l_m1.clean_loss()

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        axes[0].set_xlabel('N Epochs')
        axes[0].grid()
        axes[1].grid()
        axes[0].set_ylabel('Loss')
        axes[0].set_yscale('log')
        axes[1].set_xlabel('N Epochs')
        axes[1].set_ylabel('Metrics')
        axes[1].set_ylim([-1, 1])
        l_m2.plot(axes)
        plt.show()

        print(f'Loss on validation : {l_m2.val_loss_h[-1]} \n')
        for key, val in metric_val2.items():
            print(f'{key} : {val}  \n')

        l_m2.clean_loss()

    return model, l_m1, l_m2