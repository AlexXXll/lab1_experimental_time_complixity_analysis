import numpy as np
import datetime
import matplotlib.pyplot as plt

time_stamp_0 = datetime.datetime.now().timestamp()
N = 2000
n = np.linspace(1, N, N)
arr = np.array([])
for i in range(0, N):
    arr = np.append(arr, np.random.sample(1))
time_stamp_01 = datetime.datetime.now().timestamp()
# print("Вектор:", arr)
# print("Время создание вектора:", time_stamp_01-time_stamp_0)

runs = 5    # Количество запусков
# f(v)=const
ts1 = np.zeros(N)
for i in range(0, runs):
    ts1_ = np.array([])
    time_stamp_1 = np.array([datetime.datetime.now().timestamp()])
    for k in range(1, N + 1):
        arr1 = np.random.sample(k)
        arr1.fill(1)
        # for j in range(0, k):
        #     arr1[j] = 1
        time_stamp_1 = np.append(time_stamp_1, datetime.datetime.now().timestamp())
        ts1_ = np.append(ts1_, time_stamp_1[k] - time_stamp_1[k - 1])
    ts1 = ts1 + ts1_
ts1 = ts1 / runs
print('ts1 =', ts1)


# f(v)=SUM
ts2 = np.zeros(N)
for i in range(0, runs):
    ts2_ = np.array([])
    sum_arr2 = 0
    time_stamp_2 = np.array([datetime.datetime.now().timestamp()])
    for k in range(1, N + 1):
        arr2 = np.random.sample(k)
        sum_arr2 = np.sum(arr2)
        # for j in range(0, k):
        #     sum_arr2 = sum_arr2 + arr2[j]
        time_stamp_2 = np.append(time_stamp_2, datetime.datetime.now().timestamp())
        ts2_ = np.append(ts2_, time_stamp_2[k] - time_stamp_2[k - 1])
    ts2 = ts2 + ts2_
ts2 = ts2 / runs
print('ts2 =', ts2)


# f(v)=PROD
ts3 = np.zeros(N)
for i in range(0, runs):
    ts3_ = np.array([])
    prod_arr3 = 1
    time_stamp_3 = np.array([datetime.datetime.now().timestamp()])
    for k in range(1, N + 1):
        arr3 = np.random.sample(k)
        prod_arr3 = np.prod(arr3)
        # for j in range(0, k):
        #     prod_arr3 = prod_arr3 * arr3[j]
        time_stamp_3 = np.append(time_stamp_3, datetime.datetime.now().timestamp())
        ts3_ = np.append(ts3_, time_stamp_3[k] - time_stamp_3[k - 1])
    ts3 = ts3 + ts3_
ts3 = ts3 / runs
print('ts3 =', ts3)


# fig, ax = plt.subplots()
# ax.plot(n, ts1, label='Экспериментальное значение f(v)=1')
# ax.set_xlabel('Размерность вектора v')
# ax.set_ylabel('Время(секунды)')
# ax.legend()
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(n, ts2, label='Экспериментальное значение f(v)=SUM')
# ax.set_xlabel('Размерность вектора v')
# ax.set_ylabel('Время(секунды)')
# ax.legend()
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(n, ts3, label='Экспериментальное значение f(v)=PROD')
# ax.set_xlabel('Размерность вектора v')
# ax.set_ylabel('Время(секунды)')
# ax.legend()
# plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(3)
# fig.suptitle('Plots')
ax1.set_xlabel('Размерность вектора v')
ax1.set_ylabel('Время(секунды)')
ax2.set_xlabel('Размерность вектора v')
ax2.set_ylabel('Время(секунды)')
ax3.set_xlabel('Размерность вектора v')
ax3.set_ylabel('Время(секунды)')
ax1.plot(n, ts1, label='Экспериментальное значение f(v)=1')
ax2.plot(n, ts2, label='Экспериментальное значение f(v)=SUM')
ax3.plot(n, ts3, label='Экспериментальное значение f(v)=PROD')
ax1.legend()
ax2.legend()
ax3.legend()
plt.show()