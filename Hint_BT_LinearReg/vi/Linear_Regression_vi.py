#!/usr/bin/env python
# coding: utf-8

# # Hồi quy tuyến tính trên tập dữ liệu Boston Housing
#
# Dữ liệu này ban đầu là một phần của UCI Machine Learning Repository và hiện đã bị gỡ bỏ. Dữ liệu này cũng được đóng gói sẵn trong thư viện scikit-learn.
# Có 506 mẫu và 13 biến đặc trưng (feature) trong tập dữ liệu này. Mục tiêu là dự đoán giá nhà dựa trên các đặc trưng đã cho.
#
# Mô tả của tất cả các đặc trưng được liệt kê dưới đây:
#
#   **CRIM**: Tỷ lệ tội phạm bình quân đầu người theo từng thị trấn
#
#   **ZN**: Tỷ lệ đất ở được quy hoạch cho các lô đất trên 25.000 sq. ft (feet vuông)
#
#   **INDUS**: Tỷ lệ diện tích đất kinh doanh phi bán lẻ (mẫu Anh) trên mỗi thị trấn
#
#   **CHAS**: Biến giả sông Charles (= 1 nếu khu đất giáp sông; ngược lại = 0)
#
#   **NOX**: Nồng độ oxit nitric (phần trên 10 triệu)
#
#   **RM**: Số phòng trung bình mỗi căn nhà
#
#   **AGE**: Tỷ lệ căn nhà do chủ sở hữu ở, được xây trước năm 1940
#
#   **DIS**: Khoảng cách có trọng số đến năm trung tâm việc làm của Boston
#
#   **RAD**: Chỉ số khả năng tiếp cận các đường cao tốc hướng tâm
#
#   **TAX**: Thuế bất động sản theo giá trị đầy đủ trên mỗi $10.000
#
#   **B**: 1000(Bk - 0,63)², trong đó Bk là tỷ lệ [người gốc Phi] theo từng thị trấn
#
#   **LSTAT**: Tỷ lệ phần trăm dân số có địa vị thấp
#
#   **MEDV**: Giá trị trung vị của nhà do chủ sở hữu ở, tính theo đơn vị $1000
#
#
#

# **Import các thư viện cần thiết**

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# **Tải tập dữ liệu Boston Housing từ scikit-learn**

# In[2]:


from sklearn.datasets import load_boston

boston_dataset = load_boston()

# boston_dataset là một dictionary (từ điển)
# hãy kiểm tra xem nó chứa gì
boston_dataset.keys()


# **Nạp dữ liệu vào pandas dataframe**

# In[3]:


boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()


# **Giá trị mục tiêu (target) bị thiếu trong dữ liệu. Tạo một cột giá trị mục tiêu mới và thêm vào dataframe**

# In[4]:


boston['MEDV'] = boston_dataset.target


# **Tiền xử lý dữ liệu**

# In[5]:


# kiểm tra giá trị thiếu (missing) trong tất cả các cột
boston.isnull().sum()


# **Trực quan hóa dữ liệu**

# In[6]:


# thiết lập kích thước của hình vẽ
sns.set(rc={'figure.figsize':(10,8.27)})

# vẽ biểu đồ histogram thể hiện phân phối của giá trị mục tiêu
sns.distplot(boston['MEDV'], bins=50)
plt.show()


# **Ma trận tương quan**

# In[7]:


# tính tương quan cặp đôi cho tất cả các cột
correlation_matrix = boston.corr().round(2)


# In[8]:


# sử dụng hàm heatmap của seaborn để vẽ ma trận tương quan
# annot = True để in giá trị bên trong mỗi ô vuông
sns.heatmap(data=correlation_matrix, annot=True)


# **Nhận xét**
#
#
#
#
# *   Từ biểu đồ tương quan phía trên ta thấy **MEDV** có tương quan mạnh với **LSTAT**, **RM**
#
#
#

# In[9]:


plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')


# **Chuẩn bị dữ liệu để huấn luyện**

# In[10]:


X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']
print("X",X)


# **Chia dữ liệu thành tập huấn luyện và tập kiểm tra**

# In[11]:


from sklearn.model_selection import train_test_split

# chia tập huấn luyện và tập kiểm tra theo tỷ lệ 80% : 20%
# gán random_state bằng một giá trị bất kỳ để đảm bảo tính nhất quán (consistency)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# **Huấn luyện mô hình bằng LinearRegression của sklearn**

# In[12]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


# In[13]:


# đánh giá mô hình trên tập huấn luyện

y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("Hiệu năng của mô hình trên tập huấn luyện")
print("--------------------------------------")
print('RMSE là {}'.format(rmse))
print('Điểm R2 là {}'.format(r2))
print("\n")

# đánh giá mô hình trên tập kiểm tra

y_test_predict = lin_model.predict(X_test)
# sai số căn bậc hai trung bình bình phương của mô hình
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

# điểm R-squared của mô hình
r2 = r2_score(Y_test, y_test_predict)

print("Hiệu năng của mô hình trên tập kiểm tra")
print("--------------------------------------")
print('RMSE là {}'.format(rmse))
print('Điểm R2 là {}'.format(r2))


# In[14]:


# vẽ biểu đồ y_test so với y_pred
# lý tưởng thì nó phải là một đường thẳng
plt.scatter(Y_test, y_test_predict, marker='o')
plt.xlabel('Y_test')
plt.ylabel('y_test_predict')
plt.show()


# In[15]:


Check = np.c_[Y_test.values,y_test_predict]
print(Check)


# In[ ]:




