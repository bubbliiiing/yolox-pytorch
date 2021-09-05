#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from torchsummary import summary

from nets.yolo import YoloBody

if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m       = YoloBody(80, 's').to(device)
    
    summary(m, input_size=(3, 640, 640))
