import numpy as np
from matplotlib import pyplot as plt
import time

"""----------------------结果统计----------------------"""


def statistic(w, s_gate, xin, yin, nin, xout, yout, nout):
    wrong_cases_train = 0
    wrong_cases_test = 0

    for j in range(nin):
        if (np.dot(xin[j], w) - s_gate) * yin[j] <= 0:
            wrong_cases_train += 1
    wrong_rate_train = wrong_cases_train / nin

    for j in range(nout):
        if (np.dot(xout[j], w) - s_gate) * yout[j] <= 0:
            wrong_cases_test += 1
    wrong_rate_test = wrong_cases_test / nout

    print("训练集正确率=", 1 - wrong_rate_train)
    print("测试集正确率=", 1 - wrong_rate_test)

    return 0


"""----------------------Fisher线性判别----------------------"""


def fisher_discriminant(xin, yin):
    x = np.copy(xin)
    y = np.copy(yin)
    nx, d = np.shape(x)
    n1 = n2 = 0
    u1 = u2 = np.zeros([1, d])
    s1 = s2 = np.zeros([d, d])
    for j in range(nx):
        if y[j] == 1:
            u1 = u1 + x[j]
            n1 = n1 + 1
        else:
            u2 = u2 + x[j]
            n2 = n2 + 1
    u1 = u1 / n1
    u2 = u2 / n2
    for j in range(nx):
        if y[j] == 1:
            s1 = s1 + np.dot((x[j] - u1).T, x[j] - u1)
        else:
            s2 = s2 + np.dot((x[j] - u2).T, x[j] - u2)
    s1 = s1 / (n1 - 1)
    s2 = s2 / (n2 - 1)
    sw = s1 + s2
    sw_inverse = np.linalg.inv(sw)
    w = np.dot(sw_inverse, (u1 - u2).T)
    s_gate = np.dot(u1 + u2, w) / 2
    return w, s_gate


"""----------------------数据集初始化----------------------"""

# 数据分布与规模
mean1 = [-5, 0]
var1 = [[1, 0], [0, 1]]
mean2 = [0, 5]
var2 = [[1, 0], [0, 1]]
n = 200
train_rate = 0.8
n_train = int(n * train_rate)
n_test = n - n_train
# 数据填充
x1 = np.empty([n, 2])  # A
x2 = np.empty([n, 2])  # B
x_train = np.empty([n_train * 2, 2])  # 320
x_test = np.empty([n_test * 2, 2])  # 80

for i in range(n):  # 200
    x1[i] = np.random.multivariate_normal(mean1, var1)
    x2[i] = np.random.multivariate_normal(mean2, var2)

for i in range(n_train):  # 160
    x_train[i] = x1[i]  # A
    x_train[n_train + i] = x2[i]  # B
for i in range(n_test):  # 40
    x_test[i] = x1[i]  # A
    x_test[n_test + i] = x2[i]  # B

y_train = np.empty([n_train * 2, 1])
for i in range(n_train):
    y_train[i] = 1
    y_train[n_train + i] = -1
y_test = np.empty([n_test * 2, 1])
for i in range(n_test):
    y_test[i] = 1
    y_test[40 + i] = -1

"""----------------------代码运行----------------------"""

time_lg_start = time.time()
w_fisher, gate = fisher_discriminant(x_train, y_train)
time_lg_end = time.time()
time_lg_spend = time_lg_end - time_lg_start

x_min = min(min(x1[:, 0]), min(x2[:, 0]))
x_max = max(max(x1[:, 0]), max(x2[:, 0]))
y_min = min(min(x1[:, 1]), min(x2[:, 1]))
y_max = max(max(x1[:, 1]), max(x2[:, 1]))
x_co = np.linspace(x_min - 1, x_max + 1)

print("--------------广义逆结果统计--------------")
print("w=", w_fisher)
statistic(w_fisher, gate, x_train, y_train, n_train, x_test, y_test, n_test)
print("算法运行时间=", time_lg_spend, "s")

plt.figure("Fisher线性判别")
str1 = "fisher, x1~N(%s,%s), x2~N(%s,%s)" % (mean1, var1, mean2, var2)
plt.title(str1)
z_pla = (w_fisher[1][0] / w_fisher[0][0]) * x_co
plt.scatter(x1[:, 0], x1[:, 1], c='r')
plt.scatter(x2[:, 0], x2[:, 1], c='b')
plt.plot(x_co, z_pla, c='g')
plt.xlim(x_min - 1, x_max + 1)
plt.ylim(y_min - 1, y_max + 1)

plt.show()
