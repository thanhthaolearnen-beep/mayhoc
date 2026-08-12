#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import matplotlib.pyplot as plt


# In[24]:


# chiều cao (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# cân nặng (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
# Trực quan hóa dữ liệu
plt.plot(X, y, 'b*')
plt.axis([140, 190, 45, 75])
plt.xlabel('Chiều cao (cm)')
plt.ylabel('Cân nặng (kg)')
plt.show()


# In[25]:


# Xây dựng Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# Tính trọng số của đường thẳng khớp (fitting line)
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)
# Chuẩn bị đường thẳng khớp
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2, endpoint=True)
y0 = w_0 + w_1*x0

# Vẽ đường thẳng khớp
plt.plot(X.T, y.T, 'ro')     # dữ liệu
plt.plot(x0, y0)               # đường thẳng khớp
plt.axis([140, 190, 45, 75])
plt.xlabel('Chiều cao (cm)')
plt.ylabel('Cân nặng (kg)')
plt.show()


# In[26]:


y1 = w_1*155 + w_0
y2 = w_1*160 + w_0

print('Dự đoán cân nặng của người có chiều cao 155 cm: %.2f (kg), số liệu thật: 52 (kg)'  %(y1))
print('Dự đoán cân nặng của người có chiều cao 160 cm: %.2f (kg), số liệu thật: 56 (kg)'  %(y2))


# In[29]:


from sklearn import datasets, linear_model

# khớp mô hình bằng Hồi quy tuyến tính (Linear Regression)
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False để tính hệ số chặn (bias)
regr.fit(Xbar, y)

# So sánh hai kết quả
print('Nghiệm tìm được bằng scikit-learn  : ', regr.coef_)
print('Nghiệm tìm được từ phương trình: ', w.T)


# Trong trường hợp có nhiễu, với một cặp dữ liệu (150 cm, 70kg), kết quả đã sai

# In[11]:


import numpy as np
import matplotlib.pyplot as plt

# chiều cao (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183, 150]]).T
# cân nặng (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68, 90]]).T

# Xây dựng Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# Tính trọng số của đường thẳng khớp
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)
# Chuẩn bị đường thẳng khớp
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2, endpoint=True)
y0 = w_0 + w_1*x0

# Vẽ đường thẳng khớp
plt.plot(X, y, 'ro')     # dữ liệu
plt.plot(x0, y0)               # đường thẳng khớp
plt.axis([140, 190, 45, 95])
plt.xlabel('Chiều cao (cm)')
plt.ylabel('Cân nặng (kg)')
plt.show()


# In[ ]:




