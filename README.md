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
