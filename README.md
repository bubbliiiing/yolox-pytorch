## YOLOX：You Only Look Once目标检测模型在Pytorch当中的实现
---

## 目录
1. [性能情况 Performance](#性能情况)
2. [实现的内容 Achievement](#实现的内容)
3. [所需环境 Environment](#所需环境)
4. [小技巧的设置 TricksSet](#小技巧的设置)
5. [文件下载 Download](#文件下载)
6. [训练步骤 How2train](#训练步骤)
7. [预测步骤 How2predict](#预测步骤)
8. [评估步骤 How2eval](#评估步骤)
9. [参考资料 Reference](#Reference)

## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| COCO-Train2017 | [yolox_s.pth](https://github.com/bubbliiiing/yolox-pytorch/releases/download/v1.0/yolox_s.pth) | COCO-Val2017 | 640x640 | 38.2 | 57.7
| COCO-Train2017 | [yolox_m.pth](https://github.com/bubbliiiing/yolox-pytorch/releases/download/v1.0/yolox_m.pth) | COCO-Val2017 | 640x640 | 44.8 | 63.9
| COCO-Train2017 | [yolox_l.pth](https://github.com/bubbliiiing/yolox-pytorch/releases/download/v1.0/yolox_l.pth) | COCO-Val2017 | 640x640 | 47.9 | 66.6
| COCO-Train2017 | [yolox_x.pth](https://github.com/bubbliiiing/yolox-pytorch/releases/download/v1.0/yolox_x.pth) | COCO-Val2017 | 640x640 | 49.0 | 67.7

## 实现的内容
- [x] 主干特征提取网络：使用了Focus网络结构。  
- [x] 分类回归层：Decoupled Head，在YoloX中，Yolo Head被分为了分类回归两部分，最后预测的时候才整合在一起。
- [x] 训练用到的小技巧：Mosaic数据增强、CIOU（原版是IOU和GIOU，CIOU效果类似，都是IOU系列的，甚至更新一些）、学习率余弦退火衰减。
- [x] Anchor Free：不使用先验框
- [x] SimOTA：为不同大小的目标动态匹配正样本。

## 所需环境
pytorch==1.2.0

## 小技巧的设置
在train.py文件下：   
1、mosaic参数可用于控制是否实现Mosaic数据增强。   
2、Cosine_scheduler可用于控制是否使用学习率余弦退火衰减。   
3、label_smoothing可用于控制是否Label Smoothing平滑。

## 文件下载
训练所需的权值可在百度网盘中下载。  
链接: https://pan.baidu.com/s/1OnM-uWKETFJh_uFCAK6Vlg 提取码: b6km   

VOC数据集下载地址如下：  
VOC2007+2012训练集    
链接: https://pan.baidu.com/s/16pemiBGd-P9q2j7dZKGDFA 提取码: eiw9    

VOC2007测试集   
链接: https://pan.baidu.com/s/1BnMiFwlNwIWG9gsd4jHLig 提取码: dsda   

## 训练步骤
### a、数据集的准备
1、**本文使用VOC格式进行训练，训练前需要自己制作好数据集，如果没有自己的数据集，可以通过Github连接下载VOC12+07的数据集尝试下。**  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。    
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。    
### b、数据集的预处理 
1、训练数据集时，在model_data文件夹下建立一个cls_classes.txt，里面写所需要区分的类别。   
2、设置根目录下的voc_annotation.py里的一些参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt，即：   
```python
classes_path = 'model_data/cls_classes.txt'
```
model_data/cls_classes.txt文件内容为：     
```python
cat
dog
...
```
3、设置完成后运行voc_annotation.py，生成训练所需的2007_train.txt以及2007_val.txt。 
### c、开始网络训练  
1、通过voc_annotation.py，我们已经生成了2007_train.txt以及2007_val.txt，此时我们可以开始训练了。     
2、设置根目录下的train.py里的一些参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt，设置方式与b、数据集的预处理类似。训练自己的数据集必须要修改！    
3、设置完成后运行train.py开始训练了，在训练多个epoch后，权值会生成在logs文件夹中。   
4、训练的参数较多，大家可以在下载库后仔细看注释，其中最重要的部分依然是train.py里的classes_path。     

## d、训练结果预测
1、训练结果预测需要用到两个文件，分别是yolo.py和predict.py。   
2、设置根目录下的yolo.py里的一些参数。第一次预测可以仅修改model_path以及classes_path。训练自己的数据集必须要修改。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**   
3、设置完成后运行predict.py开始预测了，具体细节查看预测步骤。    
4、预测的参数较多，大家可以在下载库后仔细看注释，其中最重要的部分依然是yolo.py里的model_path以及classes_path。     

## 预测步骤
### a、使用预训练权重
1、下载完库后解压，在百度网盘下载各个权值，放入model_data，默认使用yolox_s.pth，其它可调整，运行predict.py，输入  
```python
img/street.jpg
```  
2、在predict.py里面进行设置可以进行video视频检测、fps测试、批量文件测试与保存。  
### b、使用自己训练的权重
1、按照训练步骤训练。  
2、在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/yolox_s.pth',
    "classes_path"      : 'model_data/coco_classes.txt',
    #---------------------------------------------------------------------#
    #   输入图片的大小，必须为32的倍数。
    #---------------------------------------------------------------------#
    "input_shape"       : [640, 640],
    #---------------------------------------------------------------------#
    #   所使用的YoloX的版本。s、m、l、x
    #---------------------------------------------------------------------#
    "phi"               : 's',
    #---------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #---------------------------------------------------------------------#
    "confidence"        : 0.5,
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    #---------------------------------------------------------------------#
    #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
    #---------------------------------------------------------------------#
    "letterbox_image"   : True,
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"              : True,
}

```
3、运行predict.py，输入   
```python
img/street.jpg
```
4、在predict.py里面进行设置可以进行video视频检测、fps测试、批量文件测试与保存。  

## 评估步骤 
1、本文使用VOC格式进行评估。     
2、划分测试集，如果在训练前已经运行过voc_annotation.py文件，代码会自动将数据集划分成训练集、验证集和测试集。  
3、如果想要修改测试集的比例，可以修改voc_annotation.py文件下的trainval_percent。trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1。train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1。   
4、设置根目录下的yolo.py里的一些参数。第一次评估可以仅修改model_path以及classes_path。训练自己的数据集必须要修改。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**     
5、设置根目录下的get_map.py里的一些参数。第一次评估可以仅修改classes_path，classes_path用于指向检测类别所对应的txt，评估自己的数据集必须要修改。与yolo.py中分开设置的原因是可以让使用者自己选择评估什么类别，而非所有类别。   
6、运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。   

## Reference
https://github.com/Megvii-BaseDetection/YOLOX
