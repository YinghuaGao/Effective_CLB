# Effective_CLB
Codes for "Not All Samples Are Born Equal: Towards Effective Clean-Label Backdoor Attacks" (PR, 2023)

## Abstract
Recent studies demonstrated that deep neural networks (DNNs) are vulnerable to backdoor attacks. The attacked model behaves normally on benign samples, while its predictions are misled whenever adversary-specified trigger patterns appear. Currently, clean-label backdoor attacks are usually regarded as the most stealthy methods in which adversaries can only poison samples from the target class with- out modifying their labels. However, these attacks can hardly succeed. In this paper, we reveal that the difficulty of clean-label attacks mainly lies in the antagonistic effects of ‘robust features’ related to the tar- get class contained in poisoned samples. Specifically, robust features tend to be easily learned by victim models and thus undermine the learning of trigger patterns. Based on these understandings, we propose a simple yet effective plug-in method to enhance clean-label backdoor attacks by poisoning ‘hard’ instead of random samples. We adopt three classical difficulty metrics as examples to implement our method. We demonstrate that our method can consistently improve vanilla attacks, based on extensive experiments on benchmark datasets.

