# Trust Prediction
This is a sample implementation of "[Power-law Distribution Aware Trust Prediction](https://zw-zhang.github.io/files/2018_IJCAI_Trust.pdf)"(IJCAI 2018).

### Requirements
```
MATLAB (MATLAB 2017a works fine for me)
```

### Usage
Run Powerlaw_TP with MATLAB
```
function [U,V,S] = Powerlaw_TP(A_input,k,beta,l1,l2,l3,l4,l5,iter,seed,use_GPU)
Inputs:
    A_input: n x n adjacency matrix
    k: dimensionality
    beta: coefficient for high-order proximity
    l1,l2,l3,l4,l5: regularization parameters
    iter: number of iterations
    seed: random seed
    use_GPU: whether to use GPU
Outputs:
    U: n x k matrix
    V: k x k matrix
    S: n x n matrix, sparse
Objective function:	
min_{U,V,S} ||(A - U * V * U' - S)||_F^2 + l1 * ||U||_F^2 + l2 * ||V||_F^2 + l3 * ||S||_F^2 + l4 * ||S||_1
```

```
### Cite
If you find this code useful, please cite our paper:
```
@inproceedings{wang2018power,
  title={Power-law Distribution Aware Trust Prediction.},
  author={Wang, Xiao and Zhang, Ziwei and Wang, Jing and Cui, Peng and Yang, Shiqiang},
  booktitle={Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence},
  pages={3564--3570},
  year={2018}
}
```