import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt



def BPNNS(x,y,step=10000,rate=0.01,debug=False):
    # check gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # process data [>0 => 1, 0 => 0]
    np.set_printoptions(threshold=np.inf)
    x_bin = np.float64(x > 0)
    x_train = x / x.sum(axis=1)[:,None]
    y_train = y  # false 1 ; true 0
    # set size
    n_in, n_h, n_out, batch_size = len(x_train[0]), 5, 1, len(x_train)
    # using GPU
    x_train = torch.tensor(x_train).cuda()
    y_train = torch.tensor(y_train.T).cuda()
    # design model
    model = nn.Sequential(nn.Linear(n_in, n_h),
                          nn.Sigmoid(),
                          nn.Linear(n_h, 1),
                          nn.Sigmoid()).double().cuda()
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
            s_f = np.multiply(s_f, x_bin[i])

    # test fail line
    model.eval()
    result = []
    for i in range(n_in):
        if (s_f[i] == 1):
            test = torch.tensor(np.float64([0] * n_in))
            test[i] = 1
            result.append((i + 1, float(model(test)[0])))
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
    return sorted_result

def run(cov_filename, result_filename):
    # print(torch.__version__)
    temp_x = np.loadtxt(cov_filename, dtype=np.float64, delimiter=",")
    temp_y = np.loadtxt(result_filename, dtype=np.float64, delimiter=",")
    return BPNNS(temp_x, temp_y,debug=False)

if __name__ == '__main__':

    #print(torch.__version__)
    temp_x = np.loadtxt("buggy1_sort_buggy1.py.csv",dtype=np.float64, delimiter=",")
    temp_y = np.loadtxt("buggy1_sort_result.txt",dtype=np.float64, delimiter=",")

    print(BPNNS(temp_x,temp_y),debug=True)