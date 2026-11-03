import numpy as np
import torch
from utils.metrics import multi_class_eval


def train_one_epoch(model, loss_func, optimizer, train_x_left, train_x_right, train_y,
                    fold_index, epoch, batch_size, device):
    train_index = np.arange(len(train_y))
    np.random.shuffle(train_index)
    step = 0
    model.train()
    for j in range(0, len(train_index), batch_size):
        index = train_index[j:j + batch_size]
        x_left = train_x_left[index]
        x_right = train_x_right[index]

        y = train_y[index].squeeze(dim=-1)
        y_pred = model(x_left, x_right)

        train_loss = loss_func(y_pred, y.to(device))

        y_pred = torch.softmax(y_pred, dim=-1).detach().cpu().numpy()
        train_acc, train_f1, train_precision, train_recall, train_kappa = multi_class_eval(
            y.numpy(),
            y_pred)
        print(
            "fold:{} epoch:{} step:{} "
            "train loss:{:.6f}, train acc:{:.3f}, train f1:{:.3f}, train precision:{:.3f}, train recall:{:.3f}, train kappa:{:.3f}"
            .format(
                fold_index, epoch, step,
                train_loss.item(), train_acc, train_f1, train_precision, train_recall, train_kappa
            ))

        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1


def test(model, loss_func, valid_x_left, valid_x_right, valid_y,
         fold_index, epoch, batch_size, device):
    model.eval()
    with torch.no_grad():
        valid_index = np.arange(len(valid_y))
        y = torch.tensor([])
        y_pred = torch.tensor([]).to(device)
        for j in range(0, len(valid_index), batch_size):
            index = valid_index[j:j + batch_size]
            x_left = valid_x_left[index]
            x_right = valid_x_right[index]
            if len(y) == 0:
                y = torch.LongTensor([])
            y = torch.concat([y, valid_y[index].squeeze(dim=-1)])
            pred = model(x_left, x_right)
            y_pred = torch.concat([y_pred, pred])

        valid_loss = loss_func(y_pred, y.to(device))

        y_pred = torch.softmax(y_pred, dim=-1).cpu().detach().numpy()
        valid_acc, valid_f1, valid_precision, valid_recall, valid_kappa = multi_class_eval(y.numpy(), y_pred)
        print(
            "fold:{} epoch:{}        "
            "valid loss:{:.6f}, valid acc:{:.3f}, valid f1:{:.3f}, valid precision:{:.3f}, valid recall:{:.3f}, valid kappa:{:.3f}"
            .format(
                fold_index, epoch,
                valid_loss.item(), valid_acc, valid_f1, valid_precision, valid_recall, valid_kappa
            ))
        score = [valid_acc, valid_f1, valid_precision, valid_recall, valid_kappa]

        return score
