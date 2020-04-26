# keras学习笔记

# 1. 将标签列表向量化

有两种方法：

​	1）转换为整数张量 

​	2）使用one-hot编码，即每一行，只有一个位置为1，其余全是0

对于损失函数的选择会有所不同：

> one-hot分类编码：loss='categorical_crossentropy'    分类交叉熵
>
> 整数张量编码：loss='sparse_categorical_crossentropy'    稀疏分类交叉熵



另外，如果只是二分类问题的话，只有两个标签，向量化更加方便

astype和训练数据的类型一致

```python
y_train = np.asarray(train_labels).astype('float32') 
y_test = np.asarray(test_labels).astype('float32')
```

## 1.1 one-hot编码

- 手动实现

```python
def to_one_hot(labels):
    dimension = len(set(labels))
    results = np.zeros((len(labels), dimension)) 
    for i, label in enumerate(labels): 
        results[i, label] = 1. 
    return results 
 
one_hot_train_labels = to_one_hot(train_labels)   
one_hot_test_labels = to_one_hot(test_labels) 
```

- 调用keras内置方法

```python
from keras.utils.np_utils import to_categorical
# 或者是：
# from keras.utils import to_categorical 
 
one_hot_train_labels = to_categorical(train_labels) 
one_hot_test_labels = to_categorical(test_labels)
```

## 1.2 转化为整数张量 

```python
y_train = np.array(train_labels, dtype='int') 
y_test = np.array(test_labels, dtype='int')
```

如果转化为整数张量， 则需要改变**损失函数**的设置

```python
model.compile(optimizer='rmsprop', 
              loss='sparse_categorical_crossentropy', 
              metrics=['acc'])
```

# 2. 减小模型过拟合的方法

也称为模型的正则化，另外一个简单有效的方法是**获取更多的训练数据**

## 2.1 减小网络大小

减少模型中可学习参数的个数（这由层数和每层的单元个数决定）

- 原始模型：

```python
from keras import models 
from keras import layers 
 
model = models.Sequential() 
model.add(layers.Dense(16, activation='relu', input_shape=(10000,))) 
model.add(layers.Dense(16, activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid'))
```

- 容量更小的模型

```python
model = models.Sequential() 
model.add(layers.Dense(4, activation='relu', input_shape=(10000,))) 
model.add(layers.Dense(4, activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid'))
```

## 2.2 添加权重正则化

奥卡姆剃刀（Occam’s razor）原理：简单模型比复杂模型更不容易过拟合。

**L1、L2**正则化，L2正则化也叫**权重衰减**

- 向模型添加L2权重正则化

  ```python
  from keras import regularizers 
   
  model = models.Sequential() 
  model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), 
                         activation='relu', input_shape=(10000,))) 
  model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), 
                         activation='relu')) 
  model.add(layers.Dense(1, activation='sigmoid'))
  ```

- 其他的权重正则化项

  ```python
  from keras import regularizers 
   
  regularizers.l1(0.001)   
   
  regularizers.l1_l2(l1=0.001, l2=0.001) 
  ```

## 2.3 添加dropout正则化

通过 Dropout 层向网络中引入 dropout，dropout 将被应用于前面一层的输出

- 向网络中添加**dropout**

  ```python
  model = models.Sequential() 
  model.add(layers.Dense(16, activation='relu', input_shape=(10000,))) 
  model.add(layers.Dropout(0.5)) 
  model.add(layers.Dense(16, activation='relu')) 
  model.add(layers.Dropout(0.5)) 
  model.add(layers.Dense(1, activation='sigmoid'))
  ```

  



























# 其他注意

- 一般来说，训练数据越少，过拟合会越严重，而较小的网络可以降低过拟合
- 网络的最后一层只有一个单元，没有激活，是一个线性层。这是标量回归（标量回归是预测单一连续值的回归）的典型设置。
- 数据集很少的情况下，一般使用**K折交叉验证**
- 测试