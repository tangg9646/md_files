# 数学笔记

## 1.向量点乘、叉乘

- 点乘，点积，内积
  - 也叫数量积。结果是一个向量在另一个向量方向上投影的长度，是一个**标量**。

- 叉乘，向量积
  - 结果是一个和已有两个向量都垂直的**向量**。
  - 该向量与这两个向量组成的坐标平面垂直

## 2.矩阵乘法

在python中，定义了数学上没有的两个矩阵 * 运算，用于实现两个矩阵对应元素的相乘

```python
aa = np.array([
    [0,1],
    [2,3]
])

bb = np.array([
    [10, 10],
    [10, 10]
])
```

```python
aa * bb
```

```python
array([[ 0, 10],
       [20, 30]])
```

数学意义上的矩阵相乘（要求第一个矩阵的列数=第二个矩阵的行数）

```python
aa.dot(bb)
```

```python
array([[10, 10],
       [50, 50]])
```



# Anaconda笔记

- 查看notebook运行在哪个python下

  ```python
  import sys
  print(sys.executable)
  ```

# GIT笔记

## 1.已有远程库，将远程库克隆到本地

- 登录GitHub，创建新仓库
- 用git clone克隆一个本地库,(在指定的某个本地文件夹下使用`git bash here`)
- 输入命令 `git clone git@github.com:tangg9646/Interview-code-practice-python.git`

## 2.先有本地仓库，后关联到远程库

现在的情景是，你已经在本地创建了一个Git仓库后，又想在GitHub创建一个Git仓库，并且让这两个仓库进行远程同步，这样，GitHub上的仓库既可以作为备份，又可以让其他人通过该仓库来协作

- 假设已有的本地仓库为桌面的 `test_git`文件夹，现在我想将其关联到远程仓库 `test_git`

  ![](https://raw.githubusercontent.com/tangg9646/my_github_image_bed/master/img20200103105826.png)

- 在本地的  `test_git`文件夹下启动 git bash here 输入以下命令

  将本地仓库和远程仓库 **关联起来**（这一步只是关联，还没进行推送和同步）

  ```powershell
  git remote add origin git@github.com:tangg9646/test_git.git
  ```

- 将本地的仓库和远程的空白仓库关联起来（将本地文件推送到远程库）

  ```powershell
  git push -u origin master
  ```

  由于远程库是空的，第一次推送`master`分支时，加上了`-u`参数，

  Git不但会把本地的`master`分支内容推送的远程新的`master`分支，还会把本地的`master`分支和远程的`master`分支关联起来，

  在以后的推送或者拉取时就可以简化命令(直接使用`git push` )。



# Python笔记

## 1. pickle保存数据

```python
import pickle

a_dict = {
    "da":123,
    2:[12,55,69,8],
    "25":{1:2, "dd":"of"}}

# 保存数据
with open("pickle_example.pickle", 'wb') as file:
    pickle.dump(a_dict, file)
   
# 读取数据

with open("pickle_example.pickle", 'rb') as file:
    a_dict2 = pickle.load(file)
    
print(a_dict2)
```

## 2. Python 的tkinter窗口

python的内置图形化窗口（GUI）

代码的复现文件夹为

**Win_Linux_Share/GUI窗口**

### 实例：计算NH3-H2流量

这个例子包括了：**标签、按钮、输入框、选择框**

标签：Label

按钮：Radiobutton

输入框：Entry

选择框：Button

代码文件下载地址：[calcu_fuel_lewis_number.py](https://github.com/tangg9646/file_share/blob/master/calcu_fuel_lewis_number.py)

程序GUI界面：

![](https://raw.githubusercontent.com/tangg9646/my_github_image_bed/master/img20200111173211.png)

## 3. 函数的参数

**可变参数**

*args类型的参数，\*args代表一个元组对象

```python
def fruits(aa, *args):
    print(aa)
    for bb in args:
        print("可变参数为：", bb)
    return
```

\**args类型的参数，**args代表一个字典对象

```python
def fruits(**args):
    for name, value in args.items():
        print("name=", name)
        print("value=", value)
        
fruits(apple=2.33, banana=4.23, orange=7.325)
```

## 4. format格式化函数

格式化字符串的函数 **str.format()**，它增强了字符串格式化的功能。

基本语法是通过 **{}** 和 **:** 来代替以前的 **%** 。

format 函数可以接受不限个参数，位置可以不按顺序。

实例：

```python
>>>"{} {}".format("hello", "world")    # 不设置指定位置，按默认顺序
'hello world'
 
>>> "{0} {1}".format("hello", "world")  # 设置指定位置
'hello world'
 
>>> "{1} {0} {1}".format("hello", "world")  # 设置指定位置
'world hello world'
```

设置参数的情况：

```python
# 通过指定名称
print("网站名：{name}, 地址 {url}".format(name="菜鸟教程", url="www.runoob.com"))
 
# 通过字典设置参数
site = {"name": "菜鸟教程", "url": "www.runoob.com"}
print("网站名：{name}, 地址 {url}".format(**site))
 
# 通过列表索引设置参数
my_list = ['菜鸟教程', 'www.runoob.com']
print("网站名：{0[0]}, 地址 {0[1]}".format(my_list))  # "0" 是必须的
```

![image.png](https://upload-images.jianshu.io/upload_images/19168686-c0dd3d733ba52a0b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```python
>>>print ("{} 对应的位置是 {{0}}".format("runoob"))
runoob 对应的位置是 {0}
```

## 5. 迭代器，生成器，装饰器

### 5.1 迭代器

可迭代对象：list , str, tiple等

迭代：通过for循环来遍历这个可迭代对象

迭代器是一个可以记住遍历未知的对象。

### 5.2 生成器

使用生成器可以生成一个值的序列用于迭代，这个值的序列并不是一次生成的，二是使用一个再生成一个，可以使程序节约大量内存。

**生成器将返回一个迭代器的函数**，并且生成器只能用于迭代操作，因此，生成器是一种特殊的迭代器。

```python
list = [
    [1,3],
    [4,6],
    [7,9]
]

def c_gene(list):
    for aa in list:
        for bb in aa:
            yield bb
            print("already finished pause")

for nn in c_gene(list):
    print(nn)
```

结果：

> 1
> already finished pause
> 3
> already finished pause
> 4
> already finished pause
> 6
> already finished pause
> 7
> already finished pause
> 9
> already finished pause



实例2

```python
# 定义一个生成器（函数）
def myYield(n):
    while n>0:
        print("\n开始生成...")
        yield n
        print("完成一次yield")
        n -= 1
        
        
for i in myYield(6):
    print("遍历得到的值为：", i)
print("\n\n")

# 注意！生成器返回的是一个迭代器的函数
# 得到生成器对象
my_yield_obj = myYield(8)
print("已经实例化生成器对象")
my_yield_obj.__next__()
print("第二次调用next方法（）")
my_yield_obj.__next__()
```

运行结果：

> ```
> 开始生成...
> 遍历得到的值为： 6
> 完成一次yield
> 
> 开始生成...
> 遍历得到的值为： 5
> 完成一次yield
> 
> 开始生成...
> 遍历得到的值为： 4
> 完成一次yield
> 
> 开始生成...
> 遍历得到的值为： 3
> 完成一次yield
> 
> 开始生成...
> 遍历得到的值为： 2
> 完成一次yield
> 
> 开始生成...
> 遍历得到的值为： 1
> 完成一次yield
> 
> 
> 
> 已经实例化生成器对象
> 
> 开始生成...
> 第二次调用next方法（）
> 完成一次yield
> 
> 开始生成...
> ```

**在深度学习的神经网络的模型训练中会用到生成器的用法**

将数据分批处理，送给模型进行训练

**【第6章 循环神经网络高级用法-温度预测问题.ipynb】**

```python
# 生成时间序列样本及其目标的生成器
def generator(data, lookback, delay, min_index, max_index, 
              shuffle=False, batch_size=128, step=6): 
    if max_index is None: 
        max_index = len(data) - delay - 1 #由于要得到预测“目标值”，因此最大值应该减去delay
    i = min_index + lookback #由于输入数据应该包含lookback个过去时间步，所以应从min_index+lookback开始
    while 1: 
        if shuffle: #如果指定打乱顺序
            rows = np.random.randint( 
                min_index + lookback, max_index, size=batch_size) 
        else: 
            if i + batch_size >= max_index: #如果剩余的数据不足以抽取一个batch_size
                i = min_index + lookback #则，将i重置为最小值
            #可供抽取的行的范围，通常是i~(i+batch_size)之间
            rows = np.arange(i, min(i + batch_size, max_index)) 
            #下一次，i从下一批次的索引开始
            i += len(rows) 
 
        #新建空白矩阵
        #（batch_size * perStep * features）
        samples = np.zeros((len(rows), 
                           lookback // step, 
                           data.shape[-1])) 
        #用于存储目标向量
        targets = np.zeros((len(rows),)) 
        
        # 正式提取样本
        for j, row in enumerate(rows): 
            indices = range(rows[j] - lookback, rows[j], step) 
            samples[j] = data[indices] 
            targets[j] = data[rows[j] + delay][1] 
        yield samples, targets
```

```python
lookback = 1440 
step = 6 
delay = 144 
batch_size = 128 
 
train_gen = generator(float_data, 
                      lookback=lookback, 
                      delay=delay, 
                      min_index=0, 
                      max_index=200000, 
                      shuffle=True, 
                      step=step, 
                      batch_size=batch_size) 
val_gen = generator(float_data, 
                    lookback=lookback, 
                    delay=delay, 
                    min_index=200001, 
                    max_index=300000, 
                    step=step, 
                    batch_size=batch_size) 
test_gen = generator(float_data, 
                     lookback=lookback, 
                     delay=delay, 
                     min_index=300001, 
                     max_index=None, 
                     step=step, 
                     batch_size=batch_size) 
 
val_steps = (300000 - 200001 - lookback)  //batch_size  
 
test_steps = (len(float_data) - 300001 - lookback)  //batch_size
```

```python
history = model.fit_generator(train_gen, 
                              steps_per_epoch=500,  
                              epochs=8,  
                              validation_data=val_gen,  
                              validation_steps=val_steps)
```

### 5.3 装饰器

用于在已经写完的长函数里面添加一些功能。

即使是不同目的或者不同类的函数或者更类，也可以插入相同的功能。



- 定义装饰器

  装饰器的定义与普通函数定义形式上相同

  装饰器函数的参数，必须要有函数或者类对象

  在装饰器函数中重新定义一个新的函数或者类，并在其中执行某些功能的前后或者中间使用被装饰的函数或类

  最后返回这个新定义的函数或者类

- 在定义的普通函数声明前一行加入

  @装饰函数名称

  使得这个普通函数被装饰器所装饰

- 最后，对被装饰的函数进行调用

#### 1) 装饰函数

实例：

```python
import functools

# 定义一个装饰器
def decorator(func):
    # 定义装饰器函数
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("进入装饰器函数内部，可以在普通函数开始执行前，添加一些新功能")
        func(*args, **kwargs)
        print("普通函数运行结束， 可以自己再添加某些功能")
    return wrapper

#装饰语句函数
@decorator
def normal_func1(x):
    a = []
    for i in range(x):
        a.append(i)
    print(a)
    
@decorator
def normal_func2(n):
    print("最喜欢的水果是：", n)
```

```python
normal_func1(5)
```

> ```
> 进入装饰器函数内部，可以在普通函数开始执行前，添加一些新功能
> [0, 1, 2, 3, 4]
> 普通函数运行结束， 可以自己再添加某些功能
> ```

```python
normal_func2("香蕉")
```

> ```
> 进入装饰器函数内部，可以在普通函数开始执行前，添加一些新功能
> 最喜欢的水果是： 香蕉
> 普通函数运行结束， 可以自己再添加某些功能
> ```

---

**如果装饰器本身需要传入参数**

---

```python
import functools

# 定义一个装饰器
def log(text):
    def decorator(func):
        
        # 定义装饰器函数
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(text)
            print("进入装饰器函数内部，可以在普通函数开始执行前，添加一些新功能")
            func(*args, **kwargs)
            print("普通函数运行结束， 可以自己再添加某些功能")

        return wrapper
    
    return decorator

#装饰语句函数
@log("这是range函数")
def normal_func1(x):
    a = []
    for i in range(x):
        a.append(i)
    print(a)
    
@log("这是喜欢的水果函数")
def normal_func2(n):
    print("最喜欢的水果是：", n)
```

```python
normal_func1(5)
```

> ```
> 这是range函数
> 进入装饰器函数内部，可以在普通函数开始执行前，添加一些新功能
> [0, 1, 2, 3, 4]
> 普通函数运行结束， 可以自己再添加某些功能
> ```

```python
normal_func2("橘子")
```

> ```
> 这是喜欢的水果函数
> 进入装饰器函数内部，可以在普通函数开始执行前，添加一些新功能
> 最喜欢的水果是： 橘子
> 普通函数运行结束， 可以自己再添加某些功能
> ```

---

**func.\__name__参数**

---

执行完装饰器参数后，normal_fun1.\__name__属性会变成了wrapper

有些依赖函数签名的代码执行就会出错，需要将其\__name__属性更换为原本的名字

解决办法：

- 导入functools模块
- 在在定义`wrapper()`的前面加上`@functools.wraps(func)`



#### 2) 装饰类

```python
# 定义一个装饰器
def decorator(myclass):
    # 定义内嵌类
    class InnerClass:
        def __init__(self, z=0):
            self.z = z
            # 实例化被装饰的类
            self.wrapper = myclass()
        def position(self):
            self.wrapper.position()
            print("z axis:", self.z)
    
    # 返回新定义的类
    return InnerClass


# 使用修饰器修饰原本的类
@decorator
class coordination:
    def __init__(self, x=100, y=100):
        self.x = x
        self.y = y
        
    def position(self):
        print("x axis:", self.x)
        print("y axis:", self.y)
```

```python
coor = coordination()
coor.position()
```

> ```python
> x axis: 100
> y axis: 100
> z axis: 0
> ```

---

**如果装饰类需要传入参数**

---

比如增加类的属性，指定属性的值

```python
# 定义一个装饰器
def add_z(zz = 99):
    # 定义装饰器
    def decorator(myclass):
        
        class InnerClass:
            def __init__(self, z=0):
                self.z = zz
                # 实例化被装饰的类
                self.wrapper = myclass()
            def position(self):
                self.wrapper.position()
                print("z axis:", self.z)
    
        # 返回新定义的类
        return InnerClass
    
    return decorator
    


# 使用修饰器修饰原本的类
@add_z(85)
class coordination:
    def __init__(self, x=100, y=100):
        self.x = x
        self.y = y
        
    def position(self):
        print("x axis:", self.x)
        print("y axis:", self.y)
```

```python
coor = coordination()
coor.position()
```

> ```
> x axis: 100
> y axis: 100
> z axis: 85
> ```

## 6. 将python程序打包成exe

通过安装pyinstaller模块，通过命名实现打包

在终端当前程序所处目录下运行：

```python
pyinstaller -w -F 你的程序.py
```

执行完成后，会在当前目录下出现三个文件：

- .spec文件：此文件无用，也可以删除。
- dist文件夹：此文件夹下有你想要的.exe文件，可以直接在命令框执行
- build文件夹：此文件夹无用，可以删除。

## 7. 模块、包 相关用法

一个.py文件就是一个模块

一个包是一个目录（文件夹），必须要有`__init___.py`文件，否则将被认为是普通目录

### 7.1 模块查找路径

　初学者会发现，自己写的模块只能在当前路径下的程序里面才能导入，换一个目录再导入自己的模块就报错了，，这是为什么呢？

　　答案就是：这与导入路径有关

```python
import sys
 
print(sys.path)
```

　　输出结果入下（这是windows下的目录）：

```python
['D:\\pycode\\模块学习', 'D:\\pycode', 'D:\\python3\\python36.zip',
 'D:\\python3\\DLLs', 'D:\\python3\\lib',
'D:\\python3',
'D:\\python3\\lib\\site-packages',
'D:\\python3\\lib\\site-packages\\win32',
'D:\\python3\\lib\\site-packages\\win32\\lib',
 'D:\\python3\\lib\\site-packages\\Pythonwin']
```

　　所以python解释器就会按照列表顺序去依次到每隔目录下去匹配你要导入的模块名，只要在一个目录下匹配到了该模块名，就立马不继续往下找，如果找完了依旧没有找到，那么就会报错。

　　（注意，可能是linux下，列表的第一个元素为空，即代表当前目录，所以你自己定义的模块在当前目录会被优先导入）

### 7.2 模块的调用

参考链接：[Python中if __name__ == '__main__'：的作用和原理](https://blog.csdn.net/heqiang525/article/details/89879056)

```python
def  myfunc()
    pass
 
mian()
```

如果直接在模块里面调用这个函数，

那么，假设其他模块需要import当前模块，那么这个程序也会执行，但是其他模块想要的只是import而已，所以显然，这样是不可取的，我们需要将这种调用方式改成下面这种。

```python
def  myfunc()
    pass

if __name__  == '__main__':
    myfunc()
```

`__name__`是当前模块名，

- 当模块被直接运行时，模块名为 `__main__`
- 当模块式被导入时， 模块名为 `model`（没有.py）

### 7.3 跨模块导入包

包的组织结构

[python 浅析模块，包及其相关用法](https://www.cnblogs.com/wj-1314/p/7510241.html)

![](https://raw.githubusercontent.com/tangg9646/my_github_image_bed/master/img20191210104040.png)

## 8. 日期处理

python有内置的日期处理模块，datetime，但是这里主要介绍pandas的日期处理功能

### 8.1 pandas中的日期基本数据类型

- Timestamp 单个日期
  - 通过 `pd.to_datetime()` 将字符串格式的日期，变成Timestamp类型
- DatetimeIndex 多个独立的日期组合起来
- PeriodIndex 指定了间隔频率的DatetimeIndex 
  - 通过将`DatetimeIndex.to_period("D")`，指定间隔频率来得到
- TimedeltaIndex 针对时间增量或者持续时间
  - 当用一个日期减去另一个日期，放回结果就是上述类型

### 8.2 有规律的时间序列

- `pd.date_range()` 

  - 指定一个时间区段，频率、自动生成这个区间

  - ```python
    pd.date_range('2019-12-8', periods=4)
    pd.date_range('2019-12-8', '2019-12-12')
    pd.date_range('2019-12-8', periods=4, freq='H')
    ```

  - 创建生成的对象为 `DatetimeIndex `类型

- `pd.period_range()`

  - 使用上述类似的方法，生成区间
  - 创建生成的对象为 `PeriodIndex`类型

- `pd.timedelta_range()`

  - 使用上述类似的方法，生成区间
  - 创建生成的对象为 `TimedeltaIndex` 类型

### 8.3 日期处理实例

用到的数据获取链接：

[https://github.com/tangg9646/file_share](https://github.com/tangg9646/file_share)

写好的jupyter notebook文件也在此仓库

```python
import numpy as np
import pandas as pd
```


```python
data = pd.read_csv("Fremont_Bridge_Bicycle_Counter.csv", index_col="Date", parse_dates=True)
data.head()
```

![](https://raw.githubusercontent.com/tangg9646/my_github_image_bed/master/img20191210154423.png)


```python
data.columns = ['Total','West', 'East']
```


```python
data.head()
```

![](https://raw.githubusercontent.com/tangg9646/my_github_image_bed/master/img20191210154509.png)


```python
data.dropna().describe()
```

![image-20191210154609507](C:\Users\TG\AppData\Roaming\Typora\typora-user-images\image-20191210154609507.png)



#### 1. 数据可视化


```python
%matplotlib inline
import seaborn
seaborn.set()
```


```python
import matplotlib.pyplot as plt
data.plot()
plt.ylabel("Hourly Bicycle Count")
```

![](https://raw.githubusercontent.com/tangg9646/my_github_image_bed/master/img20191210154633.png)




#### 2. 增大时间粒度

上述时间粒度太细，看不出趋势，重新取样成较大的时间粒度



**resample**

按周累计上述数据，重新绘图


```python
weekly = data.resample("W").sum()
weekly.plot(style=[':','--','-'])
plt.ylabel("Weekly  bicycle count")
```


![](https://raw.githubusercontent.com/tangg9646/my_github_image_bed/master/img20191210154649.png)



**数据累计平均**

另一种对数据进行累计的简便方法

计算30天的移动平均值


```python
daily = data.resample("D").sum()
```


```python
daily.rolling(30, center=True).mean().plot(style=[':','--','-'])
plt.ylabel("mean of 30 days count")
```

![](https://raw.githubusercontent.com/tangg9646/my_github_image_bed/master/img20191210154707.png)



**移动平均**

使用移动平均 来平滑曲线

高斯分布时间窗口

窗口宽度=50天
窗口内高斯平滑宽度=10天


```python
daily.rolling(50, center=True,
             win_type="gaussian").sum(std=10).plot(style=[':','--','-'])
```

![](https://raw.githubusercontent.com/tangg9646/my_github_image_bed/master/img20191210154722.png)

#### 3. 后续

后续代码不再展示，查看《python数据科学手册》P182
后面还对

- 每小时的自行车流量
- 每周每天的自行车流量
- 工作日，双休日每小时的自行车流量

做了分析，主要用到了数据透视 groupby 操作



## 9. 统计模块

scipy里面的stats包

### 9.1 寻找众数

`scipy.stats.mode`函数寻找数组或者矩阵每行/每列中最常出现成员以及出现的次数

[官方说明文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html#scipy.stats.mode)

```python
from scipy.stats import mode
def mode(a, axis=0, nan_policy='propagate'):
```

**函数作用：返回传入数组/矩阵中最常出现的成员以及出现的次数。**

**如果多个成员出现次数一样多，返回值小的那个。**

例子：

```python
list = ['a', 'a', 'a', 'b', 'b', 'b', 'a']
print("# Print mode(list):", mode(list))
print("# list中最常见的成员为：{}，出现了{}次。".format(mode(list)[0][0], mode(list)[1][0]))
 
# Print mode(list): ModeResult(mode=array(['a'], dtype='<U1'), count=array([4]))
# list中最常见的成员为：a，出现了4次。
```

```python
a = np.array([[2, 2, 2, 1],
              [1, 2, 2, 2],
              [1, 1, 3, 3]])
print("# Print mode(a):", mode(a))
print("# Print mode(a.transpose()):", mode(a.transpose()))
print("# a的每一列中最常见的成员为：{}，分别出现了{}次。".format(mode(a)[0][0], mode(a)[1][0]))
print("# a的第一列中最常见的成员为：{}，出现了{}次。".format(mode(a)[0][0][0], mode(a)[1][0][0]))
print("# a的每一行中最常见的成员为：{}，分别出现了{}次。".format(mode(a.transpose())[0][0], mode(a.transpose())[1][0]))
print("# a中最常见的成员为：{}，出现了{}次。".format(mode(a.reshape(-1))[0][0], mode(a.reshape(-1))[1][0]))
 
# a的每一列中最常见的成员为：[1 2 2 1]，分别出现了[2 2 2 1]次。
# a的第一列中最常见的成员为：1，出现了2次。
# a的每一行中最常见的成员为：[2 2 1]，分别出现了[3 3 2]次。
# a中最常见的成员为：2，出现了6次。
```

## 10. python命名约定

### 10.1 类的命名

- 在一个类的定义里，由下划线 `_` 开头的属性名（和函数名）都当做内部使用的名字，不应该在这个类之外使用。

- python对类定义里以 两个下划线开头， 但是不以两个下划线结尾的 名字，做了特殊处理

  使得在类定义之外不能直接使用这个名字访问。

### 10.2 静态方法+类方法

![](https://raw.githubusercontent.com/tangg9646/my_github_image_bed/master/img20191216102820.png)

![](https://raw.githubusercontent.com/tangg9646/my_github_image_bed/master/img20191216112439.png)

类方法的例子

![](https://raw.githubusercontent.com/tangg9646/my_github_image_bed/master/img20191216112536.png)

### 10.3 检查是否属于某个类型

![](https://raw.githubusercontent.com/tangg9646/my_github_image_bed/master/img20191216111533.png)



```python
def set_name(self, name):
    if not isinstance(name, str):
        raise PersonValueError("set_name", name)
    self._name = name
```

### 例子：学校人事管理中的类

[jupyter notebook 文件]([https://github.com/tangg9646/file_share/blob/master/%E5%AD%A6%E6%A0%A1%E4%BA%BA%E4%BA%8B%E7%AE%A1%E7%90%86%E7%B1%BB.ipynb](https://github.com/tangg9646/file_share/blob/master/学校人事管理类.ipynb))

## 11. Python文件和目录操作

参考的文章：[Python文件操作，看这篇就足够](https://juejin.im/post/5c57afb1f265da2dda6924a1#heading-6)



**路径的写法：**

```python
"C:/H2-NH3实验记录/图片处理/原始图片-视频"
```

一定要注意是**斜杠**，不是反斜杠

### 11.1 文件读写



### 11.2 获取目录下的列表

假设你当前的工作目录有一个叫 `my_directory` 的子目录，该目录包含如下内容：

![](https://raw.githubusercontent.com/tangg9646/my_github_image_bed/master/img20200113165415.png)

使用 `os.scandir()` 和 `pathlib.Path` 来获取目录下的列表

```python
import os
with os.scandir('my_directory') as entries:
    for entry in entries:
        print(entry.name)
```

```python
from pathlib import Path

entries = Path('my_directory')
for entry in entries.iterdir():
    print(entry.name)
```

都将得到以下结果：

> ```python
> file1.py
> file2.csv
> file3.txt
> sub_dir
> sub_dir_b
> sub_dir_c
> ```

总结：

![](https://raw.githubusercontent.com/tangg9646/my_github_image_bed/master/img20200113165815.png)

### 11.3 列出目录中的所有文件

过滤子目录，只列出文件

`os.scandir`方式

```python
import os

basepath = 'my_directory'
with os.scandir(basepath) as entries:
    for entry in entries:
        if entry.is_file():
            print(entry.name)
```

`pathlib.Path()`方式

```python
from pathlib import Path

basepath = Path('my_directory')
for entry in basepath.iterdir():
    if entry.is_file():
        print(entry.name)
```

如果将for循环和if语句组合成单个生成器表达式，则上述的代码可以更加简洁。

获取路径下的所有**文件**，保存至**列表** （排除掉子目录）

```
from pathlib import Path

basepath = Path('my_directory')
files_in_basepath = (entry for entry in basepath.iterdir() if entry.is_file())
for item in files_in_basepath:
    print(item.name)
```

### 11.4 列出子目录

只列出子目录，而忽视文件 

`os.scandir()`方式

```python
import os

basepath = 'my_directory'
with os.scandir(basepath) as entries:
    for entry in entries:
        if entry.is_dir():
            print(entry.name)
```

`pathlib.Path()`方式

```python
from pathlib import Path

basepath = Path('my_directory')
for entry in basepath.iterdir():
    if entry.is_dir():
        print(entry.name)
```

列表推导式

```python
basepath = Path('C:/H2-NH3实验记录/图片处理/原始图片-视频')
dirs_in_basepath = (folder for folder in basepath.iterdir() if folder.is_dir())

for item in dirs_in_basepath:
    print(item.name)
```

### 11.5 获取文件属性

获取 `my_directory` 中文件的最后修改时间。以时间戳的方式输出：

`os.scandir`方式

```python
import os

with os.scandir('C:/H2-NH3实验记录/图片处理/原始图片-视频') as entries:
    for entry in entries:
        info = entry.stat()
        print(info.st_mtime)
```

> ```python
> 1578905965.8528488
> 1578905966.991372
> 1578905045.4028094
> 1578905045.4028094
> 1578905045.4028094
> ```

`pathlib`方式

```python
from pathlib import Path

basepath = Path('C:/H2-NH3实验记录/图片处理/原始图片-视频')
for entry in basepath.iterdir():
    info = entry.stat()
    print(info.st_mtime)
```

> ```python
> 1578905965.8528488
> 1578905966.991372
> 1578905045.4028094
> 1578905045.4028094
> 1578905045.4028094
> ```

编写一个辅助函数，将时间戳转换为一个 `datetime` 对象



















