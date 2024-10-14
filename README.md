# YOLOv8_tools

## 主要功能
* 实现yolov8的半自动标注，模型支持onnx以及pt文件，支持使用GPU，支持txt、xml标签的格式互换

## 该项目主要参考以下博客
* [https://blog.csdn.net/weixin_45734021/article/details/141434727](https://blog.csdn.net/weixin_45734021/article/details/141434727)

## 环境配置：
* Python3.9
* Ultralytics 8.2.98
* Pytorch 2.4.1
* Numpy 1.26.4
* Opencv 4.10.0
* 暂不支持多GPU检测
* 详细环境配置见`requirements.txt`

## 文件结构：
```
  ├── images: 放置图片的文件夹
  ├── model: 放置模型文件的文件夹
  ├── txt: 放置txt文件的文件夹
  ├── xml: 放置xml文件的文件夹
  ├── YOLOv8_tools.py: yolov8的实用工具，包括半自动标注以及标签文件格式转换等
  ├── export_onnx.py: 将训练生成的.pt模型转换为onnx模型
```

## yolov8下载地址：
* 官网地址： [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* 百度云链接： [https://pan.baidu.com/s/1a67r_Ab9O68WXkfDI9ek-Q](https://pan.baidu.com/s/1a67r_Ab9O68WXkfDI9ek-Q)  密码: j7qb


## 使用方法
* 确保提前准备好数据
* 若要使用GPU进行半自动标注，可以直接指定想使用的GPU
* 选择要进行的任务模式`-task model2xml/txt2xml/xml2txt`

## 注意事项
* 在使用工具时，直接使用请将文件放入指定文件夹中，若想指定自己的文件路径，请将`default=""`引号中的内容更改为**绝对路径**
* 使用命令行参数时，记得先进入代码的**工作目录**
* 一定要指定使用的模式，半自动标注请使用model2xml，标签格式转换请使用txt2xml或xml2txt

## 命令行参数使用
- 如果想指定自己的图片文件、txt文件、txt文件以及模型文件进行半自动标注
- `YOLOV8_tools.py -img img_path -txt txt_path -xml xml_path -model model_path -task model2xml`
- 使用指定GPU（例如要使用第0块GPU）
- `-device 0`
- 指定模型输入图片大小（例如大小为640）
- `-size 640`

## 持续更新
* 后期可能会持续更新一些实用工具