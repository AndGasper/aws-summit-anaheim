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