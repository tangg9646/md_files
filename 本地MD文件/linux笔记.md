# linux相关知识


## 获得root权限

  ```shell
  su -
  ```

退出root状态： Ctrl+D

## linux 设置笔记本合盖不待机

  ```shell
  su -
  nano  /etc/systemd/logind.conf
  ```

  将 HandleLidSwitch 变量前的注释 # 去掉 

  修改 HandleLidSwitch 变量参数将 suspend  改为 lock 

  重启生效

  - systemctl restart systemd-login

## Jupyter Notebook 中查看当前 运行哪个python

```python
import sys
print sys.executable
```

## Anaconda Navigator 图形化界面

```powershell
anaconda-navigator
```



## 查找指定目录下的文件



# 正月点灯笼 linux入门教程

显示日期：

> date

显示日历：

> cal
>
> cal 2019
>
> cal 11 2019

目录结构：

> win 用\斜杠
>
> linux 用 / 斜杠
>
> 返回最高一级目录 cd /
>
> ​	/ 下有home目录，里面有用户
>
> ​	cd /home/tangg/
>
> ~$表示在 tangg文件夹下
>
> /$ 表示在最根一级目录下
>
> pwd 查看当前目录
>
> cd ..  返回上一级目录
>
> cd ~ 返回tangg文件夹

清除：

> clear

# 将闲置linux作为云计算端

# [如何用Linux外接显示器或投影仪](https://www.cnblogs.com/quantumman/p/4587017.html)

Screen 0: minimum 8 x 8, current 3200 x 1080, maximum 8192 x 8192
DVI-I-0 disconnected (normal left inverted right x axis y axis)
VGA-0 connected 1280x1024+0+0 (normal left inverted right x axis y axis) 376mm x 301mm
   1280x1024      60.0*+   75.0  
   1024x768       75.0     70.1     60.0  
   800x600        75.0     72.2     60.3     56.2  
   640x480        75.0     72.8     59.9  
DVI-I-1 connected 1920x1080+1280+0 (normal left inverted right x axis y axis) 509mm x 286mm
   1920x1080      60.0*+
   1680x1050      60.0  
   1440x900       59.9  
   1280x1024      75.0     60.0  
   1280x960       60.0  
   1280x720       60.0  
   1024x768       75.0     70.1     60.0  
   800x600        75.0     72.2     60.3     56.2  
   640x480        75.0     72.8     59.9  
HDMI-0 disconnected (normal left inverted right x axis y axis)





xrandr --output VGA --right-of LVDS --auto
打开外接显示器(--auto:最高分辨率)，设置为右侧扩展屏幕







