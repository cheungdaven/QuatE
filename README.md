

Hyper-parameters for reproducing the reported results are provided in the train_QuatE_dataset.py.


# How to run 
Requirements:
Pytorch 1.4+

STEP:

1.  export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/usr/local/lib64:/usr/lib64
2.  CUDA_VISIBLE_DEVICES=0 python3 train_QuatE_dataset.py



# Citation

```
@article{zhang2019quaternion,
  title={Quaternion Knowledge Graph Embedding},
  author={Zhang, Shuai and Tay, Yi and Yao, Lina and Liu, Qi},
  journal={arXiv preprint arXiv:1904.10281},
  year={2019}
}
```

This code is based on the OpenKE project.
