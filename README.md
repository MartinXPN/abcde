# abcde
Approximating Betweenness Centrality with Drop Edge


### To run the training script
```shell
# Modify the hyperparameters in `train.py` and run it
python train.py
```

### To evaluate the model
Download the evaluation data
```shell
cd datasets
./download.sh
```

Then run the prediction script on the target dataset
```shell
python predict.py --model_path experiments/latest/models/best.h5py \
                  --data_test datasets/real/amazon.txt \
                  --label_file datasets/real/amazon_score.txt
```


## Obtained results so far [v0.1.0](https://github.com/MartinXPN/abcde/releases/tag/v0.1.0) and comparison with the original [DrBC](https://github.com/FFrankyy/DrBC) paper
Obtained results on `Real` datasets (Model was run on 64GB CPU machine with 8vCPUS):

| Dataset     |  Top 1%   |  Top 5%   |   Top 10%  | Kendal Tau |  Running time  |
| ------------| :-------: | :-------: | :--------: | :--------: | :------------: |
| com-youtube | 77.2      | 76.4      | 79.0       | 59.9       | 198.0          |
| Amazon      | 92.2      | 89.4      | 87.1       | 78.3       | 412.0          |
| Dblp        | 77.3      | 69.8      | 71.3       | 72.3       | 660.4          |
| cit-Patents | Doesn't   | have the  | dataset    |    in the  | provided URL   |
| com-lj      | 69.3      | 75.9      | 78.5       | 71.7       | 2356.1         |


DrBC paper results on `Real` datasets (Model was run on an 80-core server with 512GB memory, and 8 16GB Tesla V100 GPUs. Trained on GPU, tested on only CPUs):

| Dataset     |  Top 1%   |   Top 5%  |  Top 10%   | Kendal Tau |  Running time  |
| ------------| :-------: | :-------: | :--------: | :--------: | :------------: |
| com-youtube | 73.6      | 66.7      | 69.5       | 57.3       | 402.9          |
| Amazon      | 86.2      | 79.7      | 76.9       | 69.3       | 449.8          |
| Dblp        | 78.9      | 72.0      | 72.5       | 71.9       | 566.7          |
| cit-Patents | 48.3      | 57.5      | 64.1       | 72.6       | 744.1          |
| com-lj      | 67.2      | 72.6      | 74.8       | 71.3       | 2274.2         |
