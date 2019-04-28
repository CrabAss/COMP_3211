import numpy as np


def division (a, b):
    if a == 0 or b == 0:
        return 0.0
    return a / b


def run(cov_filename, result_filename):
    test_cov = np.loadtxt(cov_filename, dtype=np.int32, delimiter=",")
    test_result = np.loadtxt(result_filename, dtype=np.int32, delimiter=",")

    test_cov_binary = np.int32(test_cov > 0)

    statement_count = len(test_cov[0])
    testcase_count = len(test_result)
    susp = [0.0] * statement_count
    total_pass = 0
    total_fail = 0
    pass_count = [0] * statement_count
    fail_count = [0] * statement_count
    
    for i in range(0, testcase_count):
        if test_result[i] == 0:
            total_pass += 1
            pass_count = np.add(test_cov_binary[i], pass_count)
        else:
            total_fail += 1
            fail_count = np.add(test_cov_binary[i],fail_count)

    for i in range(0, statement_count):
        if pass_count[i] > 0:
            susp[i] = division(division(pass_count[i], total_pass),
                               (division(pass_count[i], total_pass) + division(fail_count[i], total_fail)))
            susp[i] = 1 - susp[i]

    susp_sorted = [[idx, susp_value] for idx, susp_value in enumerate(susp, 1)]

    susp_sorted = sorted(susp_sorted, key=lambda x: x[1], reverse=True)

    return susp_sorted


if __name__ == '__main__':
    result = run("buggy_sort_buggy.py.csv", "buggy_sort_result.txt")
    print(result)
    # for idx, susp_value in result:
    #     print("line {0}, suspiciousness is {1:.3f}".format(idx, susp_value))
