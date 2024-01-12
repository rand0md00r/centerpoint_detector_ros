> **注意：** 本文件由人工智能生成，可能存在错误。在使用本文件中的信息时，请进行适当的验证。

# centerpoint_detector_ros

## 项目介绍

`centerpoint_detector_ros`是一个ROS包，它订阅点云消息，使用Centerpoint模型进行推理，
发布非地面点云和检测到的目标。

## 功能

- 订阅点云消息
- 对点云进行滤波和分割，提取地面和非地面点云
- 使用CenterPoint模型进行推理
- 发布非地面点云和检测到的目标

## 安装

首先，你需要安装ROS和一些其他的依赖项。然后，你可以按照以下步骤来安装`pcl_subscriber`：

1. 克隆仓库到你的catkin工作空间的`src`目录：

    ```bash
    cd ~/catkin_ws/src
    git clone https://github.com/your_username/pcl_subscriber.git
    ```

2. 使用catkin_make来编译：

    ```bash
    cd ~/catkin_ws
    catkin_make
    ```

3. 设置环境：

    ```bash
    source ~/catkin_ws/devel/setup.bash
    ```

## 启动项目

你可以使用以下命令来启动`pcl_subscriber`：

```bash
roslaunch pcl_subscriber run.launch
```

## 致谢

本项目的部分代码参考了https://github.com/HaohaoNJU/CenterPoint/tree/master项目。我们对原作者的贡献表示感谢。
