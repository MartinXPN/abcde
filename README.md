# abcde
Approximating Betweenness Centrality with Drop Edge


### To run the training script
```shell
# Modify the hyperparameters in `train.py` and run it
python train.py
```

### To evaluate the model
```shell
python predict.py --model_path experiments/latest/models/best.h5py \
                  --data_test datasets/Real/amazon.txt \
                  --label_file datasets/Real/amazon_score.txt
```
