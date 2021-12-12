from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy import stats
from scipy.stats.morestats import probplot

#AAA
print('A')
with open("./HW10-Polyreg.txt", "r") as f:  
    data = f.read()
list0=str(data).split()
x=[]
y=[]
for i in range(len(list0)):
    if i%2==0:
        x.append(float(list0[i]))
    else:
        y.append(float(list0[i]))
plt.title('Scatter Plot')
plt.scatter(x,y)
plt.show()
print('Not linearly')

#BBB
print('B')
mat=np.zeros((100,3))
for each in range(100):
    mat[each][0]=1
    mat[each][1]=x[each]
    mat[each][2]=(x[each])**2

y_mat=np.zeros((100,1))
for each in range(100):
    y_mat[each][0]=y[each]

mat_t=np.transpose(mat)
first=np.matmul(mat_t,mat)
ni=np.linalg.inv(first)
final=np.matmul(ni,mat_t)

B_matrix=np.matmul(final,y_mat)
print('The B matrix is:')
print(B_matrix)

#Compute SSE
Y_hat=np.matmul(mat,B_matrix)
error_mat=Y_hat-y_mat
error_mat_t=np.transpose(error_mat)
SSE=np.matmul(error_mat_t,error_mat)
print('SSE is:',SSE)
MSE=SSE/97
print('MSE is:',MSE)

covariance=MSE*ni
print('The covariance matrix is:',covariance)

#CCC
print('C')
plt.title('Residual vs Fitted')
plt.scatter(x,error_mat_t[0])
plt.show()

stats.probplot(error_mat_t[0],dist='norm',plot=plt)
plt.show()
print('Quadratic model suffice because the residuals are distrubuted around the 0 and QQ plot is approximatedly normal distribution.')

#DDD
print('D')
print('Add cubic term:')
mat=np.zeros((100,4))
for each in range(100):
    mat[each][0]=1
    mat[each][1]=x[each]
    mat[each][2]=(x[each])**2
    mat[each][3]=(x[each])**3

y_mat=np.zeros((100,1))
for each in range(100):
    y_mat[each][0]=y[each]

mat_t=np.transpose(mat)
first=np.matmul(mat_t,mat)
ni=np.linalg.inv(first)
final=np.matmul(ni,mat_t)

B_matrix=np.matmul(final,y_mat)
print('The B matrix is:')
print(B_matrix)


#Compute SSE
Y_hat=np.matmul(mat,B_matrix)
error_mat=Y_hat-y_mat
error_mat_t=np.transpose(error_mat)
SSE=np.matmul(error_mat_t,error_mat)
print('SSE is:',SSE)
MSE=SSE/96
print('MSE is:',MSE)

covariance=MSE*ni
print('The covariance matrix is:',covariance)

#EEE
print('E')
#compute P_value

test_statistic=B_matrix[3]/((MSE*covariance[3][3])**(1/2))
print('Test_statistic is:',test_statistic)
print('It follows T96 distribution.')
print('P_Value is 0.29, which is larger than 0.05, so we retain H0, and cubic term is not necessary.')


