import BPNN
import improve2_BPNN
import numpy as np
import threading
# check gpu

def re():
    for i in range(10):
        train()

def train():
    # print(torch.__version__)
    temp_x = np.loadtxt("buggy_sort_buggy.py.csv", dtype=np.float32, delimiter=",")
    temp_y = np.loadtxt("buggy_sort_result.txt", dtype=np.float32, delimiter=",")

    bugline = 7
    result = []
    imp_result = []
    for i in range(15):
        #print("round :", i + 1)
        sol = BPNN.BPNN(temp_x, temp_y,step=10000)
        sols = improve2_BPNN.BPNNS(temp_x, temp_y,step=10000)
        result.append(sol.index(bugline))
        imp_result.append(sols.index(bugline))
    #print(result)
    #print(imp_result)
    print(np.mean(result), np.mean(imp_result))

if __name__ == '__main__':

    try:
        threading.Thread(target=re).start()
        threading.Thread(target=re).start()
        threading.Thread(target=re).start()
        threading.Thread(target=re).start()
        threading.Thread(target=re).start()
    except:
        print("Error: unable to start thread")
