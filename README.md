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
# For a specific dataset
python predict.py real --model_path experiments/latest/models/best.h5py \
                       --data_test datasets/real/amazon.txt \
                       --label_file datasets/real/amazon_score.txt

# To run the whole evaluation
python predict.py all --model_path experiments/latest/models/best.h5py --datasets_dir datasets
```


## Obtained results so far [v0.1.5](https://github.com/MartinXPN/abcde/releases/tag/0.1.5) and comparison with the original [DrBC](https://github.com/FFrankyy/DrBC) paper and sampling based benchmarks
Obtained results on `Real` datasets (Model was run on 64GB CPU machine with 8vCPUS):


DrBC paper results on `Real` datasets (Model was run on an 80-core server with 512GB memory, and 8 16GB Tesla V100 GPUs. Trained on GPU, tested on only CPUs):

| **Network** |            |            | **Top-1%** |       |      |            |            |            | **Top-5%** |       |      |            |             |             | **Top-10%** |      |      |            |                 |      | **Kendall Tau** |  |      |            |            |          | **Time/s** |           |        |        |
|:-----------:|:----------:|:----------:|:-------:|:--------:|:----:|:----------:|:----------:|:----------:|:-------:|:--------:|:----:|:----------:|:-----------:|:-----------:|:-------:|:--------:|:----:|:----------:|:---------------:|:----:|:-------:|:--------:|:----:|:----------:|:----------:|:--------:|:-----------:|:--------:|:------:|:------:|
|             | ABRA       | RK         | KADABRA | Node2Vec | DrBC | ABCDE      | ABRA       | RK         | KADABRA | Node2Vec | DrBC | ABCDE      | ABRA        | RK          | KADABRA | Node2Vec | DrBC | ABCDE      | ABRA            | RK   | KADABRA | Node2Vec | DrBC | ABCDE      | ABRA       | RK       | KADABRA     | Node2Vec | DrBC   | ABCDE  |
| com-youtube | 95.7       | 76.0       | 57.5    | 12.3     | 73.6 | 77.1       | **91.2**   | 75.8       | 47.3    | 18.9     | 66.7 | 75.1       | 89.5        | **100.0**   | 44.6    | 23.6     | 69.5 | 77.6       | 56.2            | 13.9 | NA      | 46.2     | 57.3 | **59.8**   | 72898.7    | 125651.2 | **116.1**   | 4729.8   | 402.9  | 210.5  |
| amazon      | 69.2       | 86.0       | 47.6    | 16.7     | 86.2 | **92.0**   | 58.0       | 59.4       | 56.0    | 23.2     | 79.7 | **88.0**   | 60.3        | **100.0**   | 56.7    | 26.6     | 76.9 | 85.6       | 16.3            | 9.7  | NA      | 44.7     | 69.3 | **77.7**   | 5402.3     | 149680.6 | **244.7**   | 10679.0  | 449.8  | 462.7  |
| Dblp        | 49.7       | NA         | 35.2    | 11.5     | 78.9 | **79.8**   | 45.5       | NA         | 42.6    | 20.2     | 72.0 | **73.7**   | **100.0**   | NA          | 50.4    | 27.7     | 72.5 | 76.3       | 14.3            | NA   | NA      | 49.5     | 71.9 | **73.7**   | 11591.5    | NA       | **398.1**   | 17446.9  | 566.7  | 768.0  |
| cit-Patents | 37.0       | **74.4**   | 23.4    | 0.04     | 48.3 | 50.2       | 42.4       | **68.2**   | 25.1    | 0.29     | 57.5 | 58.3       | 50.9        | 53.5        | 21.6    | 0.99     | 64.1 | **64.9**   | 17.3            | 15.3 | NA      | 4.0      | 72.6 | **73.5**   | 10704.6    | 252028.5 | **568.0**   | 11729.1  | 744.1  | 1290.3 |
| com-lj      | 60.0       | 54.2*      | 31.9    | 3.9      | 67.2 | **70.9**   | 56.9       | NA         | 39.5    | 10.35    | 72.6 | **75.7**   | 63.6        | NA          | 47.6    | 15.4     | 74.8 | **78.0**   | 22.8            | NA   | NA      | 35.1     | 71.3 | **71.8**   | 34309.6    | NA       | **612.9**   | 18253.6  | 2274.2 | 2613.7 |

Bold results indivate the best performance for the given metric.
