# QuatE

Implementation of paper "Quaternion Knowledge Graph Embedding"

Hyper-parameters for reproducing the reported results are provided in the train_QuatE_dataset.py.

It seems that we don't have to normalize the relation Quaternion.

# How to run 
1.  export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:/usr/lib64
2.  CUDA_VISIBLE_DEVICES=0 python3 train_QuatE_dataset.py


NOTE: the code is based on the [OpenKE](https://github.com/thunlp/OpenKE) project.
