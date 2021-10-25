# ABCDE
ABCDE: Approximating Betweenness-Centrality ranking with progressive-DropEdge

This work was published in PeerJ Computer Science journal: https://peerj.com/articles/cs-699/

Link to the Overleaf project (PeerJ draft): https://www.overleaf.com/read/tphdqhycvwfk

### ABCDE model architecture
![](https://i.imgur.com/D0WIjO2.png)
Each Transition block is a set of {Linear → LayerNorm → PRelu → Dropout} layers, while each GCN is a set of {GCNConv → PReLU → LayerNorm → Dropout}. + symbol is the concatenation operation.
Each MaxPooling operation extracts the maximum value from the given GCN block.


### To reproduce the results
```shell
# This will run the ABCDE model on both real-world and synthetic datasets and report the results
docker run martin97/abcde:latest
```


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

*Optionally download the model reported in the paper
```shell
mkdir models && cd models
wget https://github.com/MartinXPN/abcde/releases/download/v1.0.0/best.ckpt
```

Then run the prediction script on the target dataset
```shell
# For a specific dataset
python predict.py real --model_path experiments/latest/models/best.ckpt \
                       --data_test datasets/real/amazon.txt \
                       --label_file datasets/real/amazon_score.txt

# To run the whole evaluation
python predict.py all --model_path experiments/latest/models/best.h5py --datasets_dir datasets
```

## Obtained results so far [v1.0.0](https://github.com/MartinXPN/abcde/releases/tag/v1.0.0) and comparison with the original [DrBC](https://github.com/FFrankyy/DrBC) paper and sampling based benchmarks
Obtained results on `Real` datasets (Model was run on 512GB CPU machine with 80 cores):

| Dataset     | ABRA         | RK           | KADABRA | Node2Vec | DrBC   | ABCDE     |
| :---------: | :----------: | :----------: | :-----: | :------: | :----: | :-------: |
|             |              |              | Top-1%  |          |        |           |
| com-youtube | **95.7**     | 76.0         | 57.5    | 12.3     | 73.6   | 77.1      |
| amazon      | 69.2         | 86.0         | 47.6    | 16.7     | 86.2   | **92.0**  |
| Dblp        | 49.7         | NA           | 35.2    | 11.5     | 78.9   | **79.8**  |
| cit-Patents | 37.0         | **74.4**     | 23.4    | 0.04     | 48.3   | 50.2      |
| com-lj      | 60.0         | 54.2         | 31.9    | 3.9      | 67.2   | **70.9**  |
|             |              |              | Top-5%  |          |        |           |
| com-youtube | **91.2**     | 75.8         | 47.3    | 18.9     | 66.7   | 75.1      |
| amazon      | 58.0         | 59.4         | 56.0    | 23.2     | 79.7   | **88.0**  |
| Dblp        | 45.5         | NA           | 42.6    | 20.2     | 72.0   | **73.7**  |
| cit-Patents | 42.4         | **68.2**     | 25.1    | 0.29     | 57.5   | 58.3      |
| com-lj      | 56.9         | NA           | 39.5    | 10.35    | 72.6   | **75.7**  |
|             |              |              | Top-10% |          |        |           |
| com-youtube | 89.5         | **100.0**    | 44.6    | 23.6     | 69.5   | 77.6      |
| amazon      | 60.3         | **100.0**    | 56.7    | 26.6     | 76.9   | 85.6      |
| Dblp        | **100.0**    | NA           | 50.4    | 27.7     | 72.5   | 76.3      |
| cit-Patents | 50.9         | 53.5         | 21.6    | 0.99     | 64.1   | **64.9**  |
| com-lj      | 63.6         | NA           | 47.6    | 15.4     | 74.8   | **78.0**  |
|             |              |              | Kendall Tau |      |        |           |
| com-youtube | 56.2         | 13.9         | NA      | 46.2     | 57.3   | **59.8**  |
| amazon      | 16.3         | 9.7          | NA      | 44.7     | 69.3   | **77.7**  |
| Dblp        | 14.3         | NA           | NA      | 49.5     | 71.9   | **73.7**  |
| cit-Patents | 17.3         | 15.3         | NA      | 4.0      | 72.6   | **73.5**  |
| com-lj      | 22.8         | NA           | NA      | 35.1     | 71.3   | **71.8**  |
|             |              |              | Time/s  |          |        |           |
| com-youtube | 72898.7      | 125651.2     | 116.1   | 4729.8   | 402.9  | **26.7**  |
| amazon      | 5402.3       | 149680.6     | 244.7   | 10679.0  | 449.8  | **63.5**  |
| Dblp        | 11591.5      | NA           | 398.1   | 17446.9  | 566.7  | **104.9** |
| cit-Patents | 10704.6      | 252028.5     | 568.0   | 11729.1  | 744.1  | **163.9** |
| com-lj      | 34309.6      | NA           | 612.9   | 18253.6  | 2274.2 | **271.0** |


Obtained results on `Synthetic` datasets (Model was run on 512GB CPU machine with 80 cores):

| Scale     | ABRA             | RK         | k-BC          | KADABRA   | Node2Vec   | DrBC       | ABCDE        |
| :-------: | :--------------: | :--------: | :-----------: | :-------: | :--------: | :--------: | :----------: |
|           |                  |            | Top-1%        |           |            |            |              |
| 5000      | **97.8±1.5**     | 96.8±1.7   | 94.1±0.8      | 76.2±12.5 | 19.1±4.8   | 96.5±1.8   | 97.5±1.3     |
| 10000     | **97.2±1.2**     | 96.4±1.3   | 93.3±3.1      | 74.6±16.5 | 21.2±4.3   | 96.7±1.2   | 96.9±0.9     |
| 20000     | **96.5±1.0**     | 95.5±1.1   | 91.6±4.0      | 74.6±16.7 | 16.1±3.9   | 95.6±0.9   | 96.0±1.2     |
| 50000     | **94.6±0.7**     | 93.3±0.9   | 90.1±4.7      | 73.8±14.9 | 9.6±1.3    | 92.5±1.2   | 93.6±0.9     |
| 100000    | **92.2±0.8**     | 91.5±0.8   | 88.6±4.7      | 67.0±12.4 | 9.6±1.3    | 90.3±0.9   | 91.8±0.6     |
|           |                  |            | Top-5%        |           |            |            |              |
| 5000      | 96.9±0.7         | 95.6±0.9   | 89.3±3.9      | 68.7±13.4 | 23.3±3.6   | 95.9±0.9   | **97.8±0.7** |
| 10000     | 95.6±0.8         | 94.1±0.8   | 88.4±5.1      | 70.7±13.8 | 20.5±2.7   | 95.0±0.8   | **97.0±0.6** |
| 20000     | 93.9±0.8         | 92.2±0.9   | 86.9±6.2      | 69.1±13.5 | 16.9±2.0   | 93.0±1.1   | **95.2±0.8** |
| 50000     | 90.1±0.8         | 88.0±0.8   | 84.4±7.2      | 65.8±11.7 | 13.8±1.0   | 89.2±1.1   | **92.1±0.6** |
| 100000    | 85.6±1.1         | 87.6±0.5   | 82.4±7.5      | 57.0±9.4  | 12.9±1.2   | 86.2±0.9   | **89.7±0.5** |
|           |                  |            | Top-10%       |           |            |            |              |
| 5000      | 96.1±0.7         | 94.3±0.9   | 86.7±4.5      | 67.2±12.5 | 25.4±3.4   | 94.8±0.7   | **97.6±0.4** |
| 10000     | 94.1±0.6         | 92.2±0.9   | 86.0±5.9      | 67.8±13.0 | 25.4±3.4   | 94.0±0.9   | **96.8±0.6** |
| 20000     | 92.1±0.8         | 90.6±0.9   | 84.5±6.8      | 66.1±12.4 | 19.9±1.9   | 91.9±0.9   | **94.9±0.5** |
| 50000     | 87.4±0.9         | 88.2±0.5   | 82.1±8.0      | 61.3±10.4 | 18.0±1.2   | 87.9±1.0   | **91.7±0.6** |
| 100000    | 81.8±1.5         | 87.4±0.4   | 80.1±8.2      | 52.4±8.2  | 17.3±1.3   | 85.0±0.9   | **89.4±0.5** |
|           |                  |            | Kendall Tau   |           |            |            |              |
| 5000      | 86.6±1.0         | 78.6±0.6   | 66.2±11.4     | NA        | 11.3±3.0   | 88.4±0.3   | **93.7±0.2** |
| 10000     | 81.6±1.2         | 72.3±0.6   | 67.2±13.5     | NA        | 8.5±2.3    | 86.8±0.4   | **93.3±0.1** |
| 20000     | 76.9±1.5         | 65.5±1.2   | 67.1±14.3     | NA        | 7.5±2.2    | 84.0±0.5   | **92.1±0.1** |
| 50000     | 68.2±1.3         | 53.3±1.4   | 66.2±14.1     | NA        | 7.1±1.8    | 80.1±0.5   | **90.1±0.2** |
| 100000    | 60.3±1.9         | 44.2±0.2   | 64.9±13.5     | NA        | 7.1±1.9    | 77.8±0.4   | **88.4±0.2** |
|           |                  |            | Time/s        |           |            |            |              |
| 5000      | 18.5±3.6         | 17.1±3.0   | 12.2±6.3      | 0.6±0.1   | 32.4±3.8   | **0.3±0.0** | 0.5±0.0     |
| 10000     | 29.2±4.8         | 21.0±3.6   | 47.2±27.3     | 1.0±0.2   | 73.1±7.0   | **0.6±0.0** | **0.6±0.0** |
| 20000     | 52.7±8.1         | 43.0±3.2   | 176.4±105.1   | 1.6±0.3   | 129.3±17.6 | 1.4±0.0    | **0.9±0.0**  |
| 50000     | 168.3±23.8       | 131.4±2.0  | 935.1±505.9   | 3.9±1.0   | 263.2±46.6 | 3.9±0.2    | **2.2±0.0**  |
| 100000    | 380.3±63.7       | 363.4±36.3 | 3069.2±1378.5 | 7.2±1.8   | 416.2±37.0 | 8.2±0.3    | **3.2±0.0**  |
