from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy import stats

#AAA
print('A')
with open("./HW10-PlasmaLevel.txt", "r") as f:  
    data = f.read()
list0=str(data).split()
x=[]
y=[]
for i in range(len(list0)):
    if i%2==0:
        x.append(float(list0[i]))
    else:
        y.append(float(list0[i]))


Average_x=np.mean(x)
Average_y=np.mean(y)
B1_up=0
B1_down=0
for each in range(len(x)):
    B1_up+=(x[each]-Average_x)*(y[each]-Average_y)
    B1_down+=(x[each]-Average_x)**2

B1=B1_up/B1_down

B0=Average_y-B1*Average_x

x_line=range(0,5)
y_line=B0+B1*x_line

plt.plot(x_line,y_line)

plt.scatter(x,y)

plt.ylabel('Plasma Level')
plt.xlabel('Age')
  
plt.show()

SSE=0
for each in range(len(x)):
    SSE+=(y[each]-(B0+B1*x[each]))**2

print('B0 is:',B0)
print('B1 is:',B1)
print('B matrix is [',B0,B1,']')
print('SSE is:',SSE)

error1=[]
for each in range(len(x)):
    error1.append(y[each]-(B0+B1*x[each]))
plt.scatter(x,error1)
plt.title('The residuals vs fitted value plot')
plt.show()
print('Linear model is not enough because the residuals are not evenly distributed around the 0 line')

#BBB
#BOX COX Transformation
print('B')
lambda_value=[2*(-1),1*(-1),1/2*(-1),0,1/2,1,2]
result_for_function=[]
for each in lambda_value:
    result_for_function.append(stats.boxcox_llf(each,y))
plt.plot(lambda_value,result_for_function)
plt.ylabel('log_likelihood function')
plt.xlabel('Lambda')
plt.show()
print('LogSSE=-2l/n, to make SSE small, we need to make l large')
#So we choose
print(result_for_function)
print('max function value with the lambda is: -0.5')
print("we choose lambda -0.5")

#CCC
print('C')
#New Data
new_y=[]
for each in y:
    new_y.append(each**((-1)*(1/2)))

Average_x=np.mean(x)
Average_y_new=np.mean(new_y)
B1_up=0
B1_down=0
for each in range(len(x)):
    B1_up+=(x[each]-Average_x)*(new_y[each]-Average_y_new)
    B1_down+=(x[each]-Average_x)**2

B1=B1_up/B1_down

B0=Average_y_new-B1*Average_x

x_line=range(0,5)
y_line=B0+B1*x_line
temp=[]
for each in x:
    temp.append(B0+B1*each)

plt.plot(x_line,y_line)

plt.scatter(x,new_y)

plt.ylabel('Plasma Level after Box Cox transformation')
plt.xlabel('Age')
  
plt.show()

SSE=0
for each in range(len(x)):
    SSE+=(new_y[each]-(B0+B1*x[each]))**2

print('New B0 is:',B0)
print('New B1 is:',B1)
print('New SSE is:',SSE)

#DDD
#QQ Plot
print('D')
error=[]
for each in range(len(new_y)):
    error.append(new_y[each]-temp[each])
plt.scatter(x,error)
plt.show()

stats.probplot(error,dist='norm',plot=plt)
plt.show()
print('It is a good fit because the residuals are distrubuted around the 0 and QQ plot is approximatedly normal distribution.')
        
    

