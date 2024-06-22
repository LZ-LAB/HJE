# The HJE Model
This paper has been submitted to the IEEE TKDE journal.
## Requirements
The version of Python and major packages needed to run the code:
   
    -- python 3.9.16
    -- torch(gpu) 1.12.0
    -- numpy 1.26.0
    -- tqdm 4.65.0

## How to Run
```
python main-JF17K.py                 ## JF17K dataset
python main-WikiPeople.py            ## WikiPeople dataset
python main-FB-AUTO.py               ## FB-AUTO dataset
```

## Supplementary Note
We support flexible settings of hyperparameters that have an impact on the training performance of HJE models: **num_iterations, batch_size, learning rate (lr), decay rate (dr)**. Examples of specific operations:
```
python main-FB-AUTO.py --num_iterations 1000 --batch_size 128 --lr 0.0005 --dr 0.99
```
It is worth stating that **dropout, dropout_3d, and dropout_2d** are also hyperparameters that have an impact on the final performance of the model, and to simplify the open-source code for the reader to follow, we set these four hyperparameters as fixed values in **model.py**. Nevertheless, the default values we provide for the other parameters still achieve the results reported in the paper within the error margins. In other words, in different datasets, the reader can easily better the experimental results reported in the original paper by additionally adjusting specific hyperparameter values for different **dropout, dropout_3d, and dropout_2d**.

## Citation
If you use this package for published work, please cite the following paper:
```
@article{li2024hje,
  title={HJE: Joint Convolutional Representation Learning for Knowledge Hypergraph Completion},
  author={Li, Zhao and Wang, Chenxu and Wang, Xin and Chen, Zirui and Li, Jianxin},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024},
  publisher={IEEE}
}
```
