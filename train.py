import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
# 多卡训练参考<使用教程.md>下方常见错误和解决方案
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-HSF-DETR.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
                workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                device='0,1,2,3', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
                # resume='', # last.pt path
                patience=0, # 设置0代表不早提供，设置30代表精度持续30epoch没有比之前最高的高就早停
                project='runs/train',
                name='exp',
                )