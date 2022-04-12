## 类别 5个
* _background_
* throat
* pyriform_sinus
* epiglottis
* vocal_cords_open
* vocal_cords_close
---
- 108图片有个声带闭合、梨状窝、会厌（部分）的全貌
- 梨状窝部位深，颜色暗淡
- 五个器官都在喉咙部位，和鼻咽无关
- 混有NBI照片
- 声带闭合下肯定没有喉，声带开很可能有喉
- 一幅图内多器官
- 声带和喉在同一位置
- 反光
- 分辨率高的训练策略
- 相同器官不同角度的不规则
---
* 数据准备，每个数据集分别做DataLoader
* 模型建立，直接融进来
* 模型训练,
  - 损失函数整理
  - 评估指标整理
  - 优化器选择
  - 模型保存时机
  - 训练数据记录
* 模型评估，预测，查看
---
现有模型：
`pip install segmentation-models-pytorch`
---
# 坑
* pytorch input shape 必须为 n c h w
* pytorch label shape 可以为其他格式 但是相应的loss和metrics也要变化
* num_worker=4 linux下会报错
