# Uncertainty-Guided Context Consistency Learning for Semi-supervised Semantic Segmentation
Abstract:— Semi-supervised semantic segmentation has attracted considerable attention for its ability to mitigate the reliance on extensive labeled data. However, existing consistency regularization methods only utilize high certain pixels with prediction confidence surpassing a fixed threshold for training, failing to fully leverage the potential supervisory information within the network.  Therefore, this paper proposes the uncertainty-participation  context consistency learning (\textbf{UCCL}) method to explore richer supervisory signals. Specifically, we first design the semantic backpropagation update (SBU) strategy to fully exploit the knowledge from uncertain pixel regions, enabling the model to learn consistent pixel-level semantic information from those areas. Furthermore, we propose the class-aware knowledge regulation (CKR) module to facilitate the regulation of class-level semantic features across different augmented views, promoting consistent learning of class-level semantic information within the encoder.
Experimental results on two public benchmarks demonstrate that our proposed method achieves state-of-the-art performance. 

# Pipeline
![alt text](https://github.com/YUKEKEJAN/UCCL/blob/main/Net.png)

# Installation
> pip install -r requirements.txt

# Datasets
We have demonstrated state-of-the-art experimental performance of our method on Pascal VOC2012 and Cityscapes datasets.
You can download the Pascal VOC2012 on [this](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html).

You can download the Cityscapes on [this](https://www.cityscapes-dataset.com/).

# Training 
## How to train on Pascal VOC2012
### If training is performed on the 1/2 setting, set the configuration file for the VOC dataset, set the path  for labeled data and the path  for unlabeled data, as well as the corresponding training model parameter storage path. Here is an example shell script to run UCCL on Pascal VOC2012 :

     CUDA_VISIBLE_DEVICES=0,1 nohup  python -m torch.distributed.launch --nproc_per_node=2 --master_port=6719   UCCL.py >VOC_1_2.log &

## How to train on Cityscapes
### If training is performed on the 1/2 setting, set the configuration file for the Cityscapes dataset, set the path  for labeled data and the path  for unlabeled data, as well as the corresponding training model parameter storage path. Here is an example shell script to run UCCL on Pascal VOC2012 :

     CUDA_VISIBLE_DEVICES=0,1,2,3 nohup  python -m torch.distributed.launch --nproc_per_node=4 --master_port=6719   UCCL.py >Cityscapes_1_2.log &

#  Results on Pascal VOC2012.

<img src="https://github.com/YUKEKEJAN/UCCL/blob/main/Visual.png" width="500" alt="Results on Pascal VOC2012">
<!-- ![alt text](https://github.com/YUKEKEJAN/UCCL/blob/main/Visual.png)    -->

#  Results on Cityscapes.
![alt text](https://github.com/YUKEKEJAN/UCCL/blob/main/Table2.png)   

#  Comparison of visualization results on Pascal VOC2012.
![alt text](https://github.com/YUKEKEJAN/UCCL/blob/main/Table2.png)   

