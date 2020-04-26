---
title: "1.搭建hugo个人博客"
date: 2019-11-27T17:58:59+08:00
---

本内容整理于哔哩哔哩up主，**CodeSheep**

连接如下：[手把手教你从0开始搭建自己的个人博客 |第二种姿势 | hugo](https://www.bilibili.com/video/av51574688 )

分享给想要搭建个人博客的朋友。

# 1. 常用框架

## hexo

静态的 js 博客



## jekyllrb

静态 

## hugo

基于go语言

官网：gohugo.org

## vuepress

网址：vuepress.vuejs.org

## solo

网址：solo.org



------

# 2. hugo博客搭建

本示例是在Windows平台下搭建。

过程中出现其他问题可以查看：

[https://www.gohugo.org](https://www.gohugo.org/ ) **hugo中文文档**

## 1. 安装hugo工具

- 安装Windows安装管理工具 scoop

- 安装完scoop之后，直接运行命名

  ```shell
  scoop install hugo下载博客主题
  ```

- 注意观察hugo的安装目录

**需要已经下载安装好Git到自己的电脑上**

**并且已经申请注册了github账号，可以创建自己的仓库**

推荐把git版本管理工具的基本操作学习一下：[廖雪峰Git教程](https://www.liaoxuefeng.com/wiki/896043488029600 )



## 2. 在本地创建博客目录

使用Hugo快速生成站点，比如希望生成到 `C:/Uesr/TG` 路径： 

```shell
hugo new sie C:/User/TG/tgblog
```

执行完命名后即可在对应目录下找到该文件夹

## 3. 下载hugo主题

网址为：  [huogo-themes](https://themes.gohugo.io/ )

在主题的详情页会有每个主题的git下载链接

使用命令行进入创建的tgbolg文件夹

（推荐使用git的命令行工具，**git bash**，比Windows自带的CMD好用多了呀）

```shell
cd C:/User/TG/tgblog
```

将选定好的主题，通过git命名下载到theme文件夹

```powershell
git clone https://github.com/vaga/hugo-theme-m10c.git themes/m10c
```

## 4. 在本地将hugo博客启动起来

当前所处的目录一定需要是tgblog那一层

```shell
hugo server -t m10c --buildDrafts
```

启动成功后既可以在本地1313端口（默认为此端口）访问

[http://localhost:1313](http://localhost:1313/ )

## 5. 新建一篇hugo文章

```shell
hugo new post/blog.md
```

执行完之后将会在 ' myblog/content/blog/ '目录下创建bolg.md文件

接下来就可以用第三方的软件来编写自己的博客内容

写完之后，重复步骤**4. 在本地将hugo博客启动起来**，刷新，即可在网页看到

## 6.部署到github远端仓库

部署到github，需要你首先创建一个空白的仓库，仓库的名字只能是**你的github账户名（小写）.github.io**

例如我的仓库名字为：**tangg9646.github.io**



当前处于**tgbolg**目录下

```shell
hugo --theme=m10c --baseUrl="https://tangg9646.github.io/"
```

> 注意，以上命令并不会生成草稿页面，
>
> 如果未生成任何文章，请去掉文章头部的 `draft=true` 再重新生成。）

执行 完后，会在博客目录下创建一个 **public**目录

- 把创建的public目录设置为本地的一个git仓库

  首先进入public目录

  ```shell
  git init
  git add .
  git commit -m "我的hugo博客第一次提交"
  ```

- 将本地的这个public仓库和github远端的仓库连接起来（只需要第一次的时候执行）

  同样，当前处于public目录下

  ```shell
  git remote add origin https://github.com/tangg9646/tangg9646.github.io.git
  ```

- 将本地仓库push到github仓库

  ```shell
  git push -u origin master
  ```
  
  以后再次推送到git仓库就只需要  git push 就可以完成推送

## 7. 通过公网访问博客

[https://tangg9646.github.io ](https://tangg9646.github.io/ )

根据你自己的github仓库地址而定

## 8. 主题的个性化设置

需要在下载主题的网站，选定主题的详情页一般有说明，该如何修改主题的个性化内容

### 8.1 toml简单语法

![image.png](https://upload-images.jianshu.io/upload_images/19168686-5a899d0d09a86690.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



