import os 
import numpy as np 
import matplotlib.pyplot as plt 




# rp_cifar10_low = [88.88,88.39,88.31,88.24,87.58,86.74,87.34,85.65,84.96,84.97,84.24,82.71,82.0,80.87,78.56]
# lt_cifar10_low = [88.88,89.05,89.66,89.75,89.8,89.1,89.6,89.57,89.39,89.12,88.61,88.51,88.0,87.34,86.89]

# x = np.arange(len(lt_cifar10_low))
# x_sparsity = [0.8**i for i in x]


# report_x = [x[0],x[3],x[6],x[9],x[12],x[14]]
# report_x_sparsity = [100,51.2,26.21,13.42,6.87,4.40]

# plt.plot(x, rp_cifar10_low, label='RP')
# plt.plot(x, lt_cifar10_low, label='LTH')
# plt.hlines(lt_cifar10_low[0],0,len(rp_cifar10_low), color='black', label='Dense')
# plt.xticks(report_x, report_x_sparsity, rotation=45)
# plt.xlabel('Remaining weight %')
# plt.title('CIFAR-10, ResNet-20s, low')
# plt.legend()
# plt.ylabel('Accuracy %')
# plt.savefig('acc.png')
# plt.close()






overlap_dense_sparse = np.loadtxt('overlap_21.txt')
rp = overlap_dense_sparse[:14]
lt = overlap_dense_sparse[14:]


x = np.arange(1, 15)
x_sparsity = [0.8**i for i in x]

report_x = [1,3,5,7,9,11,13]
report_x_sparsity = [64.00,40.96,26.21,16.78,10.74,6.87,4.40]



plt.plot(rp, label='RP')
plt.plot(lt, label='LTH')
plt.legend()
plt.xticks(report_x, report_x_sparsity)
plt.xlabel('Remaining weight %')
plt.ylabel('Gradient-Overlap')
# plt.title('g(Sparse), H(Dense)')
# plt.savefig('dense_sparse.png')
# plt.close()


plt.title('H(Sparse), g(Dense)')
plt.savefig('sparse_dense.png')
plt.close()

