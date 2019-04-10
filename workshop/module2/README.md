# Module 2
- Overview of Amazon SageMaker
- Amazon SageMaker DeepAR
    - LSTM-parameterized




1. Collect and prepare training data
2. Choose and optimize ML algorithm 
3. Set up and manage environments for  testing
4. Train and tune model
5. Deploy model in production
6. Scale and manage production environment



VPC
IAM
AutoScaling Group



Amazon Sagemaker DeepaR

DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks 
Valentin Flunkert, David Salinas, and Jan Gasthunas (sp?)


Setting up the data pipeline
Cleaning and processing the data
Training the model
Updating the dataset
Etc. 

context_length =>  how much context of the time series the model should take into account when making the prediction (how many previous points to look at)

prediction_length => a typical value to start with for context_length


prediction_length = units
context_length = units

prediction_length = int(7*24/int(freq.replace('H', '')))
context_length = int(7*24/int(freq.replace('H', '')))


```
prediction_length = int(7*24/int(freq.replace('H','')))
context_length = int(7*24/int(freq.replace('H','')))

print('Preditction length: %i %s' %(prediction_length, freq))
print('Context length: %i %s' %(context_length, freq))
print('-->  1 week')

n_weeks = 6
end_training = df.index[-n_weeks*prediction_length]

time_series = []
for ts in df.columns:
    time_series.append(df[ts])
    
time_series_training = []
for ts in df.columns:
    time_series_training.append(df.loc[:end_training][ts])
```


import json

def series_to_obj(ts, cat=None):
    obj = {"start": str(ts.index[0]), "target": list(ts)}
    if cat:
        obj["cat"] = cat
    return obj

def series_to_jsonline(ts, cat=None):
    return json.dumps(series_to_obj(ts, cat))

encoding = "utf-8"
s3filesystem = s3fs.S3FileSystem()

with s3filesystem.open(s3_data_path + "/train/train.json", 'wb') as fp:
    for ts in time_series_training:
        fp.write(series_to_jsonline(ts).encode(encoding))
        fp.write('\n'.encode(encoding))

with s3filesystem.open(s3_data_path + "/test/test.json", 'wb') as fp:
    for ts in time_series:
        fp.write(series_to_jsonline(ts).encode(encoding))
        fp.write('\n'.encode(encoding))





```
2019-04-10 20:56:18 Starting - Starting the training job...
2019-04-10 20:56:20 Starting - Launching requested ML instances......
2019-04-10 20:57:24 Starting - Preparing the instances for training......
2019-04-10 20:58:30 Downloading - Downloading input data..
Arguments: train
[04/10/2019 20:58:57 INFO 140605409826624] Reading default configuration from /opt/amazon/lib/python2.7/site-packages/algorithm/default-input.json: {u'num_dynamic_feat': u'auto', u'dropout_rate': u'0.10', u'mini_batch_size': u'128', u'test_quantiles': u'[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]', u'_tuning_objective_metric': u'', u'_num_gpus': u'auto', u'num_eval_samples': u'100', u'learning_rate': u'0.001', u'num_cells': u'40', u'num_layers': u'2', u'embedding_dimension': u'10', u'_kvstore': u'auto', u'_num_kv_servers': u'auto', u'cardinality': u'auto', u'likelihood': u'student-t', u'early_stopping_patience': u''}
[04/10/2019 20:58:57 INFO 140605409826624] Reading provided configuration from /opt/ml/input/config/hyperparameters.json: {u'dropout_rate': u'0.05', u'learning_rate': u'0.001', u'num_cells': u'40', u'prediction_length': u'84', u'epochs': u'100', u'time_freq': u'2H', u'context_length': u'84', u'num_layers': u'3', u'mini_batch_size': u'32', u'likelihood': u'gaussian', u'early_stopping_patience': u'10'}
[04/10/2019 20:58:57 INFO 140605409826624] Final configuration: {u'dropout_rate': u'0.05', u'test_quantiles': u'[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]', u'_tuning_objective_metric': u'', u'num_eval_samples': u'100', u'learning_rate': u'0.001', u'num_layers': u'3', u'epochs': u'100', u'embedding_dimension': u'10', u'num_cells': u'40', u'_num_kv_servers': u'auto', u'mini_batch_size': u'32', u'likelihood': u'gaussian', u'num_dynamic_feat': u'auto', u'cardinality': u'auto', u'_num_gpus': u'auto', u'prediction_length': u'84', u'time_freq': u'2H', u'context_length': u'84', u'_kvstore': u'auto', u'early_stopping_patience': u'10'}
Process 1 is a worker.
[04/10/2019 20:58:57 INFO 140605409826624] Detected entry point for worker worker
[04/10/2019 20:58:57 INFO 140605409826624] Using early stopping with patience 10
[04/10/2019 20:58:57 INFO 140605409826624] [cardinality=auto] `cat` field was NOT found in the file `/opt/ml/input/data/train/train.json` and will NOT be used for training.
[04/10/2019 20:58:57 INFO 140605409826624] [num_dynamic_feat=auto] `dynamic_feat` field was NOT found in the file `/opt/ml/input/data/train/train.json` and will NOT be used for training.
[04/10/2019 20:58:57 INFO 140605409826624] Training set statistics:
[04/10/2019 20:58:57 INFO 140605409826624] Real time series
[04/10/2019 20:58:57 INFO 140605409826624] number of time series: 7
[04/10/2019 20:58:57 INFO 140605409826624] number of observations: 117544
[04/10/2019 20:58:57 INFO 140605409826624] mean target length: 16792
[04/10/2019 20:58:57 INFO 140605409826624] min/mean/max target: 0.0/4323.66806752/30116.3691406
[04/10/2019 20:58:57 INFO 140605409826624] mean abs(target): 4323.66806752
[04/10/2019 20:58:57 INFO 140605409826624] contains missing values: no
[04/10/2019 20:58:57 INFO 140605409826624] Small number of time series. Doing 10 number of passes over dataset per epoch.
[04/10/2019 20:58:57 INFO 140605409826624] Test set statistics:
[04/10/2019 20:58:57 INFO 140605409826624] Real time series
[04/10/2019 20:58:57 INFO 140605409826624] number of time series: 7
[04/10/2019 20:58:57 INFO 140605409826624] number of observations: 121065
[04/10/2019 20:58:57 INFO 140605409826624] mean target length: 17295
[04/10/2019 20:58:57 INFO 140605409826624] min/mean/max target: 0.0/4325.92199748/30116.3691406
[04/10/2019 20:58:57 INFO 140605409826624] mean abs(target): 4325.92199748
[04/10/2019 20:58:57 INFO 140605409826624] contains missing values: no
[04/10/2019 20:58:57 INFO 140605409826624] nvidia-smi took: 0.0251798629761 secs to identify 0 gpus
[04/10/2019 20:58:57 INFO 140605409826624] Number of GPUs being used: 0
[04/10/2019 20:58:57 INFO 140605409826624] Create Store: local
#metrics {"Metrics": {"get_graph.time": {"count": 1, "max": 3844.56205368042, "sum": 3844.56205368042, "min": 3844.56205368042}}, "EndTime": 1554929941.113722, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929937.268248}

[04/10/2019 20:59:01 INFO 140605409826624] Number of GPUs being used: 0
#metrics {"Metrics": {"initialize.time": {"count": 1, "max": 4674.616098403931, "sum": 4674.616098403931, "min": 4674.616098403931}}, "EndTime": 1554929941.943111, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929941.113865}

[04/10/2019 20:59:02 INFO 140605409826624] Epoch[0] Batch[0] avg_epoch_loss=9.568491
[04/10/2019 20:59:03 INFO 140605409826624] Epoch[0] Batch[5] avg_epoch_loss=9.446704
[04/10/2019 20:59:03 INFO 140605409826624] Epoch[0] Batch [5]#011Speed: 169.46 samples/sec#011loss=9.446704
[04/10/2019 20:59:04 INFO 140605409826624] processed a total of 302 examples
#metrics {"Metrics": {"epochs": {"count": 1, "max": 100, "sum": 100.0, "min": 100}, "update.time": {"count": 1, "max": 2521.4619636535645, "sum": 2521.4619636535645, "min": 2521.4619636535645}}, "EndTime": 1554929944.464703, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929941.943172}

[04/10/2019 20:59:04 INFO 140605409826624] #throughput_metric: host=algo-1, train throughput=119.765896158 records/second
[04/10/2019 20:59:04 INFO 140605409826624] #progress_metric: host=algo-1, completed 1 % of epochs
[04/10/2019 20:59:04 INFO 140605409826624] best epoch loss so far
[04/10/2019 20:59:04 INFO 140605409826624] Saved checkpoint to "/opt/ml/model/state_67c58247-1d09-4ecb-b4a9-751749f9a7ce-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 57.66582489013672, "sum": 57.66582489013672, "min": 57.66582489013672}}, "EndTime": 1554929944.522875, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929944.464792}

[04/10/2019 20:59:05 INFO 140605409826624] Epoch[1] Batch[0] avg_epoch_loss=7.832370
[04/10/2019 20:59:06 INFO 140605409826624] Epoch[1] Batch[5] avg_epoch_loss=7.859599
[04/10/2019 20:59:06 INFO 140605409826624] Epoch[1] Batch [5]#011Speed: 174.45 samples/sec#011loss=7.859599

2019-04-10 20:58:54 Training - Training image download completed. Training in progress.[04/10/2019 20:59:06 INFO 140605409826624] Epoch[1] Batch[10] avg_epoch_loss=7.665974
[04/10/2019 20:59:06 INFO 140605409826624] Epoch[1] Batch [10]#011Speed: 162.93 samples/sec#011loss=7.433623
[04/10/2019 20:59:06 INFO 140605409826624] processed a total of 344 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2471.5700149536133, "sum": 2471.5700149536133, "min": 2471.5700149536133}}, "EndTime": 1554929946.994587, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929944.522949}

[04/10/2019 20:59:06 INFO 140605409826624] #throughput_metric: host=algo-1, train throughput=139.176260987 records/second
[04/10/2019 20:59:06 INFO 140605409826624] #progress_metric: host=algo-1, completed 2 % of epochs
[04/10/2019 20:59:06 INFO 140605409826624] best epoch loss so far
[04/10/2019 20:59:07 INFO 140605409826624] Saved checkpoint to "/opt/ml/model/state_8c1833cf-c33b-4390-8ddc-b5a7a2c4bc0f-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 56.31303787231445, "sum": 56.31303787231445, "min": 56.31303787231445}}, "EndTime": 1554929947.051357, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929946.994663}

[04/10/2019 20:59:07 INFO 140605409826624] Epoch[2] Batch[0] avg_epoch_loss=7.816970
[04/10/2019 20:59:08 INFO 140605409826624] Epoch[2] Batch[5] avg_epoch_loss=8.507609
[04/10/2019 20:59:08 INFO 140605409826624] Epoch[2] Batch [5]#011Speed: 169.82 samples/sec#011loss=8.507609
[04/10/2019 20:59:09 INFO 140605409826624] processed a total of 317 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2433.210849761963, "sum": 2433.210849761963, "min": 2433.210849761963}}, "EndTime": 1554929949.484701, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929947.05143}

[04/10/2019 20:59:09 INFO 140605409826624] #throughput_metric: host=algo-1, train throughput=130.274160663 records/second
[04/10/2019 20:59:09 INFO 140605409826624] #progress_metric: host=algo-1, completed 3 % of epochs
[04/10/2019 20:59:09 INFO 140605409826624] loss did not improve
[04/10/2019 20:59:10 INFO 140605409826624] Epoch[3] Batch[0] avg_epoch_loss=19.746582
[04/10/2019 20:59:11 INFO 140605409826624] Epoch[3] Batch[5] avg_epoch_loss=9.608880
[04/10/2019 20:59:11 INFO 140605409826624] Epoch[3] Batch [5]#011Speed: 162.54 samples/sec#011loss=9.608880
[04/10/2019 20:59:11 INFO 140605409826624] processed a total of 288 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2094.133138656616, "sum": 2094.133138656616, "min": 2094.133138656616}}, "EndTime": 1554929951.579249, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929949.484784}

[04/10/2019 20:59:11 INFO 140605409826624] #throughput_metric: host=algo-1, train throughput=137.519854752 records/second
[04/10/2019 20:59:11 INFO 140605409826624] #progress_metric: host=algo-1, completed 4 % of epochs
[04/10/2019 20:59:11 INFO 140605409826624] loss did not improve
[04/10/2019 20:59:12 INFO 140605409826624] Epoch[4] Batch[0] avg_epoch_loss=7.222675
[04/10/2019 20:59:13 INFO 140605409826624] Epoch[4] Batch[5] avg_epoch_loss=7.049080
[04/10/2019 20:59:13 INFO 140605409826624] Epoch[4] Batch [5]#011Speed: 172.20 samples/sec#011loss=7.049080
[04/10/2019 20:59:14 INFO 140605409826624] Epoch[4] Batch[10] avg_epoch_loss=7.236684
[04/10/2019 20:59:14 INFO 140605409826624] Epoch[4] Batch [10]#011Speed: 174.46 samples/sec#011loss=7.461809
[04/10/2019 20:59:14 INFO 140605409826624] processed a total of 352 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2476.3128757476807, "sum": 2476.3128757476807, "min": 2476.3128757476807}}, "EndTime": 1554929954.055947, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929951.579325}

[04/10/2019 20:59:14 INFO 140605409826624] #throughput_metric: host=algo-1, train throughput=142.13952557 records/second
[04/10/2019 20:59:14 INFO 140605409826624] #progress_metric: host=algo-1, completed 5 % of epochs
[04/10/2019 20:59:14 INFO 140605409826624] best epoch loss so far
[04/10/2019 20:59:14 INFO 140605409826624] Saved checkpoint to "/opt/ml/model/state_a556b127-161e-493c-8e10-a5801b09659a-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 77.71897315979004, "sum": 77.71897315979004, "min": 77.71897315979004}}, "EndTime": 1554929954.134117, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929954.056039}

[04/10/2019 20:59:14 INFO 140605409826624] Epoch[5] Batch[0] avg_epoch_loss=7.625322
[04/10/2019 20:59:15 INFO 140605409826624] Epoch[5] Batch[5] avg_epoch_loss=7.187330
[04/10/2019 20:59:15 INFO 140605409826624] Epoch[5] Batch [5]#011Speed: 163.75 samples/sec#011loss=7.187330
[04/10/2019 20:59:16 INFO 140605409826624] processed a total of 319 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2326.9898891448975, "sum": 2326.9898891448975, "min": 2326.9898891448975}}, "EndTime": 1554929956.461221, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929954.134168}

[04/10/2019 20:59:16 INFO 140605409826624] #throughput_metric: host=algo-1, train throughput=137.080259464 records/second
[04/10/2019 20:59:16 INFO 140605409826624] #progress_metric: host=algo-1, completed 6 % of epochs
[04/10/2019 20:59:16 INFO 140605409826624] best epoch loss so far
[04/10/2019 20:59:16 INFO 140605409826624] Saved checkpoint to "/opt/ml/model/state_07e38371-e483-4c20-9e5f-b53a5197cb13-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 65.60206413269043, "sum": 65.60206413269043, "min": 65.60206413269043}}, "EndTime": 1554929956.527266, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929956.461299}

[04/10/2019 20:59:17 INFO 140605409826624] Epoch[6] Batch[0] avg_epoch_loss=7.207187
[04/10/2019 20:59:18 INFO 140605409826624] Epoch[6] Batch[5] avg_epoch_loss=7.230352
[04/10/2019 20:59:18 INFO 140605409826624] Epoch[6] Batch [5]#011Speed: 175.47 samples/sec#011loss=7.230352
[04/10/2019 20:59:18 INFO 140605409826624] Epoch[6] Batch[10] avg_epoch_loss=7.118376
[04/10/2019 20:59:18 INFO 140605409826624] Epoch[6] Batch [10]#011Speed: 168.88 samples/sec#011loss=6.984004
[04/10/2019 20:59:18 INFO 140605409826624] processed a total of 342 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2435.6319904327393, "sum": 2435.6319904327393, "min": 2435.6319904327393}}, "EndTime": 1554929958.963023, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929956.52733}

[04/10/2019 20:59:18 INFO 140605409826624] #throughput_metric: host=algo-1, train throughput=140.409363822 records/second
[04/10/2019 20:59:18 INFO 140605409826624] #progress_metric: host=algo-1, completed 7 % of epochs
[04/10/2019 20:59:18 INFO 140605409826624] best epoch loss so far
[04/10/2019 20:59:19 INFO 140605409826624] Saved checkpoint to "/opt/ml/model/state_fc6ceec6-112b-4772-ad53-168d490cbe03-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 86.4408016204834, "sum": 86.4408016204834, "min": 86.4408016204834}}, "EndTime": 1554929959.049879, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929958.963091}

[04/10/2019 20:59:19 INFO 140605409826624] Epoch[7] Batch[0] avg_epoch_loss=6.875841
[04/10/2019 20:59:20 INFO 140605409826624] Epoch[7] Batch[5] avg_epoch_loss=7.193214
[04/10/2019 20:59:20 INFO 140605409826624] Epoch[7] Batch [5]#011Speed: 172.61 samples/sec#011loss=7.193214
[04/10/2019 20:59:21 INFO 140605409826624] Epoch[7] Batch[10] avg_epoch_loss=7.102148
[04/10/2019 20:59:21 INFO 140605409826624] Epoch[7] Batch [10]#011Speed: 164.97 samples/sec#011loss=6.992870
[04/10/2019 20:59:21 INFO 140605409826624] processed a total of 325 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2466.5958881378174, "sum": 2466.5958881378174, "min": 2466.5958881378174}}, "EndTime": 1554929961.516607, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929959.049952}

[04/10/2019 20:59:21 INFO 140605409826624] #throughput_metric: host=algo-1, train throughput=131.754819351 records/second
[04/10/2019 20:59:21 INFO 140605409826624] #progress_metric: host=algo-1, completed 8 % of epochs
[04/10/2019 20:59:21 INFO 140605409826624] best epoch loss so far
[04/10/2019 20:59:21 INFO 140605409826624] Saved checkpoint to "/opt/ml/model/state_66780c61-d979-4a68-b6b7-9356e0addb89-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 88.2120132446289, "sum": 88.2120132446289, "min": 88.2120132446289}}, "EndTime": 1554929961.605278, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929961.516678}

[04/10/2019 20:59:22 INFO 140605409826624] Epoch[8] Batch[0] avg_epoch_loss=6.464474
[04/10/2019 20:59:23 INFO 140605409826624] Epoch[8] Batch[5] avg_epoch_loss=7.126670
[04/10/2019 20:59:23 INFO 140605409826624] Epoch[8] Batch [5]#011Speed: 166.52 samples/sec#011loss=7.126670
[04/10/2019 20:59:23 INFO 140605409826624] processed a total of 315 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2299.4489669799805, "sum": 2299.4489669799805, "min": 2299.4489669799805}}, "EndTime": 1554929963.90486, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929961.60535}

[04/10/2019 20:59:23 INFO 140605409826624] #throughput_metric: host=algo-1, train throughput=136.981302735 records/second
[04/10/2019 20:59:23 INFO 140605409826624] #progress_metric: host=algo-1, completed 9 % of epochs
[04/10/2019 20:59:23 INFO 140605409826624] best epoch loss so far
[04/10/2019 20:59:23 INFO 140605409826624] Saved checkpoint to "/opt/ml/model/state_0de53a38-2a97-415e-97c3-3e640800c942-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 69.5030689239502, "sum": 69.5030689239502, "min": 69.5030689239502}}, "EndTime": 1554929963.974819, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929963.904958}

[04/10/2019 20:59:24 INFO 140605409826624] Epoch[9] Batch[0] avg_epoch_loss=7.089527
[04/10/2019 20:59:25 INFO 140605409826624] Epoch[9] Batch[5] avg_epoch_loss=7.466397
[04/10/2019 20:59:25 INFO 140605409826624] Epoch[9] Batch [5]#011Speed: 167.06 samples/sec#011loss=7.466397
[04/10/2019 20:59:26 INFO 140605409826624] Epoch[9] Batch[10] avg_epoch_loss=7.502941
[04/10/2019 20:59:26 INFO 140605409826624] Epoch[9] Batch [10]#011Speed: 169.41 samples/sec#011loss=7.546795
[04/10/2019 20:59:26 INFO 140605409826624] processed a total of 338 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2467.745065689087, "sum": 2467.745065689087, "min": 2467.745065689087}}, "EndTime": 1554929966.442699, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929963.974894}

[04/10/2019 20:59:26 INFO 140605409826624] #throughput_metric: host=algo-1, train throughput=136.961046414 records/second
[04/10/2019 20:59:26 INFO 140605409826624] #progress_metric: host=algo-1, completed 10 % of epochs
[04/10/2019 20:59:26 INFO 140605409826624] loss did not improve
[04/10/2019 20:59:27 INFO 140605409826624] Epoch[10] Batch[0] avg_epoch_loss=6.483825
[04/10/2019 20:59:27 INFO 140605409826624] Epoch[10] Batch[5] avg_epoch_loss=7.087127
[04/10/2019 20:59:27 INFO 140605409826624] Epoch[10] Batch [5]#011Speed: 171.45 samples/sec#011loss=7.087127
[04/10/2019 20:59:28 INFO 140605409826624] processed a total of 303 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2282.4249267578125, "sum": 2282.4249267578125, "min": 2282.4249267578125}}, "EndTime": 1554929968.7255, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929966.442774}

[04/10/2019 20:59:28 INFO 140605409826624] #throughput_metric: host=algo-1, train throughput=132.747027624 records/second
[04/10/2019 20:59:28 INFO 140605409826624] #progress_metric: host=algo-1, completed 11 % of epochs
[04/10/2019 20:59:28 INFO 140605409826624] best epoch loss so far
[04/10/2019 20:59:28 INFO 140605409826624] Saved checkpoint to "/opt/ml/model/state_d7a1158c-0b64-4b70-885f-994c31f94d2c-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 65.62113761901855, "sum": 65.62113761901855, "min": 65.62113761901855}}, "EndTime": 1554929968.791556, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929968.725576}

[04/10/2019 20:59:29 INFO 140605409826624] Epoch[11] Batch[0] avg_epoch_loss=6.790092
[04/10/2019 20:59:30 INFO 140605409826624] Epoch[11] Batch[5] avg_epoch_loss=7.013624
[04/10/2019 20:59:30 INFO 140605409826624] Epoch[11] Batch [5]#011Speed: 175.49 samples/sec#011loss=7.013624
[04/10/2019 20:59:31 INFO 140605409826624] Epoch[11] Batch[10] avg_epoch_loss=7.067254
[04/10/2019 20:59:31 INFO 140605409826624] Epoch[11] Batch [10]#011Speed: 166.30 samples/sec#011loss=7.131609
[04/10/2019 20:59:31 INFO 140605409826624] processed a total of 327 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2469.696044921875, "sum": 2469.696044921875, "min": 2469.696044921875}}, "EndTime": 1554929971.261394, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929968.791631}

[04/10/2019 20:59:31 INFO 140605409826624] #throughput_metric: host=algo-1, train throughput=132.398528491 records/second
[04/10/2019 20:59:31 INFO 140605409826624] #progress_metric: host=algo-1, completed 12 % of epochs
[04/10/2019 20:59:31 INFO 140605409826624] loss did not improve
[04/10/2019 20:59:31 INFO 140605409826624] Epoch[12] Batch[0] avg_epoch_loss=7.170490
[04/10/2019 20:59:32 INFO 140605409826624] Epoch[12] Batch[5] avg_epoch_loss=7.143809
[04/10/2019 20:59:32 INFO 140605409826624] Epoch[12] Batch [5]#011Speed: 180.62 samples/sec#011loss=7.143809
[04/10/2019 20:59:33 INFO 140605409826624] processed a total of 285 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 1955.6050300598145, "sum": 1955.6050300598145, "min": 1955.6050300598145}}, "EndTime": 1554929973.21741, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929971.261475}

[04/10/2019 20:59:33 INFO 140605409826624] #throughput_metric: host=algo-1, train throughput=145.726918948 records/second
[04/10/2019 20:59:33 INFO 140605409826624] #progress_metric: host=algo-1, completed 13 % of epochs
[04/10/2019 20:59:33 INFO 140605409826624] loss did not improve
[04/10/2019 20:59:33 INFO 140605409826624] Epoch[13] Batch[0] avg_epoch_loss=7.439252
[04/10/2019 20:59:34 INFO 140605409826624] Epoch[13] Batch[5] avg_epoch_loss=7.125231
[04/10/2019 20:59:34 INFO 140605409826624] Epoch[13] Batch [5]#011Speed: 173.84 samples/sec#011loss=7.125231
[04/10/2019 20:59:35 INFO 140605409826624] Epoch[13] Batch[10] avg_epoch_loss=7.139412
[04/10/2019 20:59:35 INFO 140605409826624] Epoch[13] Batch [10]#011Speed: 166.56 samples/sec#011loss=7.156430
[04/10/2019 20:59:35 INFO 140605409826624] processed a total of 341 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2432.224988937378, "sum": 2432.224988937378, "min": 2432.224988937378}}, "EndTime": 1554929975.650037, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929973.21748}

[04/10/2019 20:59:35 INFO 140605409826624] #throughput_metric: host=algo-1, train throughput=140.194730098 records/second
[04/10/2019 20:59:35 INFO 140605409826624] #progress_metric: host=algo-1, completed 14 % of epochs
[04/10/2019 20:59:35 INFO 140605409826624] loss did not improve
[04/10/2019 20:59:36 INFO 140605409826624] Epoch[14] Batch[0] avg_epoch_loss=6.431362
[04/10/2019 20:59:37 INFO 140605409826624] Epoch[14] Batch[5] avg_epoch_loss=7.034916
[04/10/2019 20:59:37 INFO 140605409826624] Epoch[14] Batch [5]#011Speed: 175.07 samples/sec#011loss=7.034916
[04/10/2019 20:59:38 INFO 140605409826624] Epoch[14] Batch[10] avg_epoch_loss=7.035146
[04/10/2019 20:59:38 INFO 140605409826624] Epoch[14] Batch [10]#011Speed: 173.85 samples/sec#011loss=7.035421
[04/10/2019 20:59:38 INFO 140605409826624] processed a total of 345 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2431.0691356658936, "sum": 2431.0691356658936, "min": 2431.0691356658936}}, "EndTime": 1554929978.081507, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929975.650108}

[04/10/2019 20:59:38 INFO 140605409826624] #throughput_metric: host=algo-1, train throughput=141.905633949 records/second
[04/10/2019 20:59:38 INFO 140605409826624] #progress_metric: host=algo-1, completed 15 % of epochs
[04/10/2019 20:59:38 INFO 140605409826624] loss did not improve
[04/10/2019 20:59:38 INFO 140605409826624] Epoch[15] Batch[0] avg_epoch_loss=7.270087
[04/10/2019 20:59:39 INFO 140605409826624] Epoch[15] Batch[5] avg_epoch_loss=7.179111
[04/10/2019 20:59:39 INFO 140605409826624] Epoch[15] Batch [5]#011Speed: 178.19 samples/sec#011loss=7.179111
[04/10/2019 20:59:40 INFO 140605409826624] Epoch[15] Batch[10] avg_epoch_loss=7.022899
[04/10/2019 20:59:40 INFO 140605409826624] Epoch[15] Batch [10]#011Speed: 171.42 samples/sec#011loss=6.835445
[04/10/2019 20:59:40 INFO 140605409826624] processed a total of 342 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2385.791063308716, "sum": 2385.791063308716, "min": 2385.791063308716}}, "EndTime": 1554929980.467726, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929978.081593}

[04/10/2019 20:59:40 INFO 140605409826624] #throughput_metric: host=algo-1, train throughput=143.341775709 records/second
[04/10/2019 20:59:40 INFO 140605409826624] #progress_metric: host=algo-1, completed 16 % of epochs
[04/10/2019 20:59:40 INFO 140605409826624] loss did not improve
[04/10/2019 20:59:41 INFO 140605409826624] Epoch[16] Batch[0] avg_epoch_loss=6.886700
[04/10/2019 20:59:41 INFO 140605409826624] Epoch[16] Batch[5] avg_epoch_loss=6.821039
[04/10/2019 20:59:41 INFO 140605409826624] Epoch[16] Batch [5]#011Speed: 176.66 samples/sec#011loss=6.821039
[04/10/2019 20:59:42 INFO 140605409826624] Epoch[16] Batch[10] avg_epoch_loss=6.774076
[04/10/2019 20:59:42 INFO 140605409826624] Epoch[16] Batch [10]#011Speed: 174.99 samples/sec#011loss=6.717720
[04/10/2019 20:59:42 INFO 140605409826624] processed a total of 343 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2396.209955215454, "sum": 2396.209955215454, "min": 2396.209955215454}}, "EndTime": 1554929982.864323, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929980.467804}

[04/10/2019 20:59:42 INFO 140605409826624] #throughput_metric: host=algo-1, train throughput=143.136620037 records/second
[04/10/2019 20:59:42 INFO 140605409826624] #progress_metric: host=algo-1, completed 17 % of epochs
[04/10/2019 20:59:42 INFO 140605409826624] best epoch loss so far
[04/10/2019 20:59:42 INFO 140605409826624] Saved checkpoint to "/opt/ml/model/state_51e5e6a6-284e-44c0-ac7c-9777e24b4ffc-0000.params"
#metrics {"Metrics": {"state.serialize.time": {"count": 1, "max": 56.730031967163086, "sum": 56.730031967163086, "min": 56.730031967163086}}, "EndTime": 1554929982.921504, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929982.864387}

[04/10/2019 20:59:43 INFO 140605409826624] Epoch[17] Batch[0] avg_epoch_loss=6.619168
[04/10/2019 20:59:44 INFO 140605409826624] Epoch[17] Batch[5] avg_epoch_loss=6.937174
[04/10/2019 20:59:44 INFO 140605409826624] Epoch[17] Batch [5]#011Speed: 171.92 samples/sec#011loss=6.937174
[04/10/2019 20:59:45 INFO 140605409826624] Epoch[17] Batch[10] avg_epoch_loss=6.912727
[04/10/2019 20:59:45 INFO 140605409826624] Epoch[17] Batch [10]#011Speed: 169.04 samples/sec#011loss=6.883391
[04/10/2019 20:59:45 INFO 140605409826624] processed a total of 337 examples
#metrics {"Metrics": {"update.time": {"count": 1, "max": 2431.8931102752686, "sum": 2431.8931102752686, "min": 2431.8931102752686}}, "EndTime": 1554929985.353529, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "AWS/DeepAR"}, "StartTime": 1554929982.921578}

[04/10/2019 20:59:45 INFO 140605409826624] #throughput_metric: host=algo-1, train throughput=138.569083403 records/second
[04/10/2019 20:59:45 INFO 140605409826624] #progress_metric: host=algo-1, completed 18 % of epochs
[04/10/2019 20:59:45 INFO 140605409826624] loss did not improve
[04/10/2019 20:59:45 INFO 140605409826624] Epoch[18] Batch[0] avg_epoch_loss=6.849690
[04/10/2019 20:59:46 INFO 140605409826624] Epoch[18] Batch[5] avg_epoch_loss=7.032488
[04/10/2019 20:59:46 INFO 140605409826624] Epoch[18] Batch [5]#011Speed: 171.37 samples/sec#011loss=7.032488
```