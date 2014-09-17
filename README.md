Minimum Probability Flow learning (MPF)
================================

MPF is a technique for parameter estimation in un-normalized probabilistic models. It is described in the paper:
> J Sohl-Dickstein, P Battaglino, MR DeWeese<br>
> Minimum probability flow learning<br>
> International Conference on Machine Learning (2011)<br>
> http://arxiv.org/abs/0906.4779

This repository contains Matlab code implementing MPF for the Ising model and the RBM.  The directory structure is as follows:
- **MPF_ising/** - parameter estimation in the Ising model
- **MPF_RBM_compare_log_likelihood/** - parameter estimation in
        Restricted Boltzmann Machines. This directory also includes
        code comparing the log likelihood of small RBMs trained via
        pseudolikelihood and Contrastive Divergence to ones trained
        via MPF.

If you're interesting in using MPF to build an Ising model of neural spike data, you should also check out Liberty Hamilton's repository at https://github.com/libertyh/ising-model.
