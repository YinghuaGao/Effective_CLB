# Effective_CLB
This repository contains code for the paper "Not All Samples Are Born Equal: Towards Effective Clean-Label Backdoor Attacks" (PR, 2023)

## Abstract
Recent studies demonstrated that deep neural networks (DNNs) are vulnerable to backdoor attacks. The attacked model behaves normally on benign samples, while its predictions are misled whenever adversary-specified trigger patterns appear. Currently, clean-label backdoor attacks are usually regarded as the most stealthy methods in which adversaries can only poison samples from the target class with- out modifying their labels. However, these attacks can hardly succeed. In this paper, we reveal that the difficulty of clean-label attacks mainly lies in the antagonistic effects of ‘robust features’ related to the tar- get class contained in poisoned samples. Specifically, robust features tend to be easily learned by victim models and thus undermine the learning of trigger patterns. Based on these understandings, we propose a simple yet effective plug-in method to enhance clean-label backdoor attacks by poisoning ‘hard’ instead of random samples. We adopt three classical difficulty metrics as examples to implement our method. We demonstrate that our method can consistently improve vanilla attacks, based on extensive experiments on benchmark datasets.

## Requirements
* python = 3.7.13
* numpy = 1.21.5
* torch = 1.12.1
* torchvision = 0.2.1

## A Quick Start
**Step 1: Calculate three metrics (loss value, gradient norm and forggeting events)**

```
CUDA_VISIBLE_DEVICES=0 python cal_metric.py --output_dir save_metric
```

**Step 2: Train backdoored model with different sample selection methods**

```
CUDA_VISIBLE_DEVICES=0 python train_backdoor.py --output_dir save_metric --result_dir save_res --backdoor_type badnets --y_target 0 --select_epoch 10
```

`--select epoch` specifies the epoch which to calculate the statistics. `--output_dir` must be the same with the one used in cal_metric.py

## Citing
If you use our code, please cite our work

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

If you have any questions, please contact us: 

yh-gao18@mails.tsinghua.edu.cn



## Acknowledgement
Our implementation is based on the following projects. We thank the authors for releasing the code.

* [An Empirical Study of Example Forgetting during Deep Neural Network Learning](https://github.com/mtoneva/example_forgetting)
* [Adversarial Neuron Pruning Purifies Backdoored Deep Models](https://github.com/csdongxian/ANP_backdoor)

