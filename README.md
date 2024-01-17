# Effective_CLB
This is the official implementation of our paper [Not All Samples Are Born Equal: Towards Effective Clean-Label Backdoor Attacks](https://www.sciencedirect.com/science/article/abs/pii/S0031320323002121), accepted by Pattern Recognition (2023). This research project is developed based on Python 3 and Pytorch, created by [Yinghua Gao]() and [Yiming Li](http://liyiming.tech/).



## Requirements
* python = 3.7.13
* numpy = 1.21.5
* torch = 1.12.1
* torchvision = 0.2.1

## A Quick Start
**Step 1: Calculate three metrics (loss value, gradient norm and forgetting events)**

```
CUDA_VISIBLE_DEVICES=0 python cal_metric.py --output_dir save_metric
```

**Step 2: Train backdoored model with different sample selection methods**

```
CUDA_VISIBLE_DEVICES=0 python train_backdoor.py --output_dir save_metric --result_dir save_res --selection forget --backdoor_type badnets --y_target 0 --select_epoch 10
```

`--select epoch` specifies the epoch used to calculate the statistics. `--output_dir` must be the same with the one used in cal_metric.py

## Citing
If this work or our codes are useful for your research, please kindly cite our paper as follows.

```
@article{gao2023not,
  title={Not all samples are born equal: Towards effective clean-label backdoor attacks},
  author={Gao, Yinghua and Li, Yiming and Zhu, Linghui and Wu, Dongxian and Jiang, Yong and Xia, Shu-Tao},
  journal={Pattern Recognition},
  volume={139},
  pages={109512},
  year={2023},
  publisher={Elsevier}
}
```




## Acknowledgement
Our implementation is based on the following projects. We sincerely thank the authors for releasing their codes.

* [An Empirical Study of Example Forgetting during Deep Neural Network Learning](https://github.com/mtoneva/example_forgetting)
* [Adversarial Neuron Pruning Purifies Backdoored Deep Models](https://github.com/csdongxian/ANP_backdoor)

