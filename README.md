# CRNN_mapping_baseline
&#8195;&#8195;基于mapping的CRNN复现

&#8195;&#8195;使用数据集：TIMIT + 四种噪声 [下载](baidu.com)

&#8195;&#8195;本次实验为复现论文 “A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement“，是基于mapping的语音增强实验。

* #### 项目结构

  ```
  - load_data				          # 数据预处理
  - net					              # 网络定义
  - recovery			              # 恢复语音
  - test			                  # 测试模块
  - utils			          
  - logs					           # tensorboard 日志文件
  ```