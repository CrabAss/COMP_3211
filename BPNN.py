import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# check gpu
device = torch.device('cpu')


def run(cov_filename, result_filename):
    # print(torch.__version__)
    temp_x = np.loadtxt(cov_filename, dtype=np.float32, delimiter=",")
    temp_y = np.loadtxt(result_filename, dtype=np.float32, delimiter=",")
    return BPNN(temp_x, temp_y)


def BPNN(x,y,step=5000,rate=0.01,debug=False):
    # process data [>0 => 1, 0 => 0]
    x_train = np.float32(x > 0)
    y_train = y  # false 1 ; true 0
    # set size
    n_in, n_h, n_out, batch_size = len(x_train[0]), 5, 1, len(x_train)
    # using GPU
    x_train = torch.tensor(x_train).cuda()
    y_train = torch.tensor(y_train.T).cuda()
    # design model
    model = nn.Sequential(nn.Linear(n_in, n_h),
                          nn.Sigmoid(),
                          nn.Linear(n_h, n_out),
                          nn.Sigmoid()).cuda()
    # loss
    criterion = torch.nn.MSELoss()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=rate, weight_decay=1e-6)

    # use for plot
    loss_list = []

    # train
    for i in range(1, step+1):
        # forward prapagation
        y_pred = model(x_train)

        # loss
        loss = criterion(y_pred, y_train)
        loss_list.append(float(loss))

        # print(model.named_parameters())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if debug and i % 1000 == 0:
            print('i: ', i, ' loss: ', loss.item())

    x_train = x_train.cpu()
    model = model.cpu()
    # print loss result
    if debug:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.plot([i for i in range(step)], loss_list, c='r')
        plt.show()

    # get fail line set
    s_f = [1] * n_in
    for i in range(batch_size):
        if (y_train[i] == 1):
            s_f = np.multiply(s_f, x_train[i])
    # test fail line
    model.eval()
    result = []
    for i in range(n_in):


        if (s_f[i] == 1):
            test = torch.tensor(np.float32([0] * n_in))
            test[i] = 1
            result.append((i + 1, float(model(test)[0])))
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
    return sorted_result


if __name__ == '__main__':
    result = run("buggy_sort_buggy.py.csv", "buggy_sort_result.txt")
    print(result)
