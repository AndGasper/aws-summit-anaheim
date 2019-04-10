#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'workshop\module4'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Lab 4: Using LSTNet with Amazon SageMaker to Build a Custom Forecasting Model
# 
# ## Overview of Lab
# 
# In this lab, you will use LSTNet with Amazon SageMaker to build, train and host a state of the art time series forecasting model.
# 
# ## Dataset Information and License
# 
# For this lab, you will be using an open source dataset entitled [“Individual Household Electric Power Consumption”](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption) that comes from the UCI Machine Learning Repository. Information about the dataset license can be found below.
# 
# The MIT License (MIT) Copyright © [2017] Zalando SE, [https://tech.zalando.com](https://tech.zalando.com)
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
# The other dataset that will be used in this lab is the [MNIST Database of handwritten digits](http://yann.lecun.com/exdb/mnist/).
# 
# ## Lab Instructions
# 
# To complete this lab, carefully move through this notebook, from top to bottom, making sure to read all text instructions/explanations and run each code cell in order. Also be sure to view the code cell outputs. To run each cell, step-by-step in the Jupyter notebook, click within the cell and press **SHIFT + ENTER** or choose **Run** at the top of the page. You will know that a code cell has completed running when you see a number inside the square brackets located to the left of the code cell. Alternatively, **[ ]** indicates the cell has yet to be run, and **[*]** indicates that the cell is still processing.
# 
# ## Table of Contents <a name="toc"></a>
# 1. <a href="#section1">Section 1: Porting LSTNet to Amazon SageMaker</a>
# 1. <a href="#section2">Section 2: LSTNet Distributed Training</a>
# 1. <a href="#section3">Section 3: Challenge - Deploying an Endpoint Using Amazon SageMaker</a>
# 
# ## Section 1: Porting LSTNet to Amazon SageMaker
# <a name="section1"></a>
# 
# In this section, you will port the LSTNet model to be trained using Amazon Sagemaker.
# 
# An LSTNet model has already been developed. There are several modules containing supporting classes stored in Amazon S3:
# 
# 1. `lstnet.py`
#   * The declaration of the model and forward function.
#   * The model consists of a convolutional layer, dropout, a GRU, a Skip GRU, a fully connected layer, and the parallel autoregresive component.
# 
# 2. `timeseriesdataset.py`
#   * Classes for loading the data: `TimeSeriesData` and `TimeSeriesDataset`.
#   * `TimeSeriesDataset` is a subclass of `gluon.data.Dataset`.
#   * It implements the `__getitem__` function which returns a time series for the given index.
#   * These classes are used to load an input file and to generate successive examples with a specified window and horizon.
#   * The window is the length of timeseries used as input data for the prediction, and the horizon is the number of time steps between the end of the window and the time at which the prediction is for.
# 
# 3. `lstnet_sagemaker.py`
#   * This module implements the `train()` function which is used as the entrypoint for training the model on a server.
#   * This is called by Amazon SageMaker on each host in the training cluster.
# <div style="text-align: right"><a href="#toc">Back to top</a></div>

#%%
import boto3
import os
import sagemaker
import pandas as pd
import numpy as np

from mxnet import gluon
from mxnet.gluon.utils import download

from sagemaker import get_execution_role
from sagemaker.mxnet import MXNet
from sagemaker.mxnet.model import MXNetModel

from IPython.display import HTML

role = get_execution_role()

#%% [markdown]
# ### Environment Variables
# 
# Configure the following variables for your environment:
# 
# 1. `bucket` - The bucket name to be used to store the training data and model artifacts.
#   * Replace *LAB_BUCKET* with the value of the **LabBucket** output in the Qwiklabs console.
# 
# 2. `prefix` - The folder name which is used inside the bucket.
#   * This can be left as 'lstnet'.

#%%
bucket = 'LAB_BUCKET'

prefix = 'lstnet'

test_bucket_prefix = '/test/'
single_host_train_bucket_prefix = '/train/single_host/'

data_dir = './data'
data_file_path = os.path.join(data_dir,'electricity.txt')

#%% [markdown]
# ### Load the Data
# 
# The next step is to load the electricity dataset from the `electricity.txt` file that was downloaded to the notebook automatically.
# 
# * The data is normalised so each reading is between 0 and 1. This is done by dividing each column by the maximum value of the column. A column is an electricity consumption time series for a single customer.
# 
# There are 321 time series of electricity consumption with 26,304 time periods in each. 

#%%
df = pd.read_csv(data_file_path,header=None)
max_columns = df.max().astype(np.float64)

# Normalize
df = df/max_columns
print(df.shape)

#%% [markdown]
# ### Split Training, Test and Upload to S3
# 
# The first 80% of the time series data is used for training and the last 20% is used as a test set.
# 
# These datasets are written to a CSV file and then uploaded to Amazon S3 to be used in training.

#%%
train_frac = 0.8

num_time_steps = len(df)
split_index = int(num_time_steps*train_frac)
train = df[0:split_index]
print('Training size {}'.format(len(train)))
test = df[split_index:]
print('Test size {}'.format(len(test)))

test_file_path = os.path.join(data_dir,'test.csv')
test.to_csv(test_file_path,header=None,index=False)
train_file_path = os.path.join(data_dir,'train.csv')
train.to_csv(train_file_path,header=None,index=False)

client = boto3.client('s3')
client.upload_file(test_file_path, bucket, prefix + test_bucket_prefix + 'test.csv')
client.upload_file(train_file_path, bucket, prefix + single_host_train_bucket_prefix + 'train.csv')

#%% [markdown]
# ### Test Locally
# 
# To make sure there are no obvious bugs in the code, use the `train()` function to test locally within your notebook. This is done with 1 epoch to verify that it executed correctly. There are also some basic unit tests included in the lab.
# 
# The key parameters to the `train()` function in this case are:
# 
#   - `hyperparameters`: The Amazon SageMaker Hyperparameters dictionary. A dictionary of string-to-string maps.
#   - `channel_input_dirs`: A dictionary of string-to-string maps from the Amazon SageMaker algorithm input channel name to the directory containing files for that input channel.
#     - **Note:** If the Amazon SageMaker training job is run in `PIPE` mode, this dictionary will be empty.
#   - `output_data_dir`: The Amazon SageMaker output data directory. After the function returns, data written to this directory is made available in the Amazon SageMaker training job output location.
#   - `num_gpus`: The number of GPU devices available on the host this script is being executed on. As we are running on a CPU notebook instance, this number is set to zero.
#   - `num_cpus`: The number of CPU devices available on the host this script is being executed on.
#   - `hosts`: A list of hostnames in the Amazon SageMaker training job cluster.
#   - `current_host`: This host's name.
#     - It will exist in the `hosts` list as well.
#   - `kwargs`: Other keyword arguments.
#   
# **Note:** You can ignore any warnings outputted by the below code cell. On an `ml.m4.xlarge` notebook this cell takes approximately 7 minutes to complete so feel free to skip this cell and move on to the next code block. During real experimentation, you may use a larger instance type to speed up testing.

#%%
from lstnet_sagemaker import train
hyperparameters = {
    'conv_hid' : 10,
    'gru_hid' : 10,
    'skip_gru_hid' : 2,
    'skip' : 5,
    'ar_window' : 6,
    'window' : 24*7,
    'horizon' : 24,
    'learning_rate' : 0.01,
    'clip_gradient' : 10.,
    'batch_size' : 128,
    'epochs' : 1
}
channel_input_dirs = {
    'train':data_dir,
    'test':data_dir
}
train(hyperparameters = hyperparameters,
      input_data_config = None,
      channel_input_dirs = channel_input_dirs,
      output_data_dir = os.path.join(data_dir, 'output'),
      model_dir = None,
      num_gpus = 0,
      num_cpus = 1,
      hosts = ['localhost'],
      current_host = 'localhost')

#%% [markdown]
# ### Choose Hyperparameters
# 
# Below, a set of reasonable hyperparameters are chosen.
# 
# The current number of epochs is set to 10 which takes approximately 5 minutes to run on a `ml.m4.xlarge` instance. A real world training job may take place over hundreds of epochs and could run for hours depending on complexity and compute power available.
# 
# **Challenge:** Can you tweak these to make the network converge faster with a lower rmse?

#%%
hyperparameters = {
    'conv_hid' : 100,
    'gru_hid' : 100,
    'skip_gru_hid' : 5,
    'skip' : 24,
    'ar_window' : 24,
    'window' : 24*7,
    'horizon' : 24,
    'learning_rate' : 0.001,
    'clip_gradient' : 10.,
    'batch_size' : 64,
    'epochs' : 10
}

#%% [markdown]
# ### Trigger the Training Job Using the Amazon SageMaker Python SDK
# 
# The final step is to trigger the training job using the high-level Python SDK. A lower-level SDK is also available for more detailed control of the parameters.
# 
# First, an estimator is created with `sagemaker.mxnet.MXNet`. The inputs are:
# 
#   * `entry_point='lstnet_sagemaker.py'`: The module used to run the training by calling the `train()` function.
#   * `source_dir='.'`: An optional directory containing code which is copied onto the Amazon SageMaker training hosts and made available to the training script.
#   * `role=role`: The IAM role which is given to the training hosts giving them privileges such as access to the S3 bucket.
#   * `output_path='s3://{}/{}/output'.format(bucket, prefix)`: The Amazon S3 bucket to store artifacts such as the model parameters.
#   * `train_instance_count=1`: The number of hosts used for training.
#     * Using a number greater than 1 will start a cluster.
#     * To take advantage of this, the training data should be sharded.
#   * `train_instance_type='ml.p3.2xlarge'` The Amazon EC2 instance type to be used for training hosts.
#     * In this case the latest generation accelerated instance, `p3`, is chosen with a Nvidia Tesla v100 GPU.
#   * `hyperparameters=hyperparameters`: The hyperparameter dictionary made available to the `train()` function in the endpoint script.
# 
# Then, the `fit()` method of the estimator is called. The parameters for this method are:
# 
#   * `inputs`: A dictionary containing the URLs in S3 of the 'train' data directory and the 'test' data directory.
#   * `wait` - This is specified as `False` so the `fit()` method returns immediately after the training job is created.
#     * Go to the Amazon SageMaker console to monitor the progress of the job.
#     * Set `wait` to `True` to block and see the progress of the training job output in the notebook itself.

#%%
lstnet1 = MXNet(entry_point='lstnet_sagemaker.py',
                source_dir='.',
                role=role,
                output_path='s3://{}/{}/output'.format(bucket, prefix),
                train_instance_count=1,
                train_instance_type='ml.p3.2xlarge',
                hyperparameters=hyperparameters, 
                framework_version=1.2)

lstnet1.fit(inputs={
                'train': 's3://{}/{}{}'.format(bucket, prefix, single_host_train_bucket_prefix),
                'test': 's3://{}/{}{}'.format(bucket, prefix, test_bucket_prefix)
            },
            wait=False)

#%% [markdown]
# **Note:** `wait` is set to `False`, this means that you will not see output from the training job in the notebook as it progresses. To view this you need to access the logs in CloudWatch via the Training Job section of the Amazon SageMaker console. By setting wait to False, you are able to initiate the training job and continue through the notebook without waiting for it to complete. The job takes approximately 9 minutes to complete with training duration of 6 minutes.
# 
# Once you initiate the training job, wait until you see an output above which says `INFO:sagemaker:Creating training-job with name: sagemaker-mxnet-XXXXXXX` where XXXXXXX is a set of numbers relating to the date and time the job was created. When you see this output, you can follow the training job progress via the Amazon SageMaker console in the "Training Jobs" section.
# 
# 
# 
# **Challenge:** 
# 1. Review the training job logs in CloudWatch and identify the hyperparameters used. How long does one epoch take to complete?
# 2. Graph the GPU utilisation in CloudWatch to identify how efficient the training job is. What might reduce efficiency when running a training job?
# 
# 
# ### Section Complete
# 
# You now have successfully ported LSTNet to Amazon SageMaker. The next step is to modify it to run across multiple hosts to train faster.
# 
# ## Section 2: LSTNet Distributed Training
# <a name="section2"></a>
# 
# In this section, the LSTNet model which has been ported to use Amazon SageMaker is modified to be run with distributed training.
# 
# ### Overview
# 
# There are three main steps required to scale the training using multiple GPUs and multiple hosts (for the purposes of this lab we will be using a single GPU and multiple CPU hosts):
# 
# 1. Pass the appropriate `kvstore` parameter to the Gluon trainer.
#   * This specifies how parameters are synchronised between batches.
#   * In this case, `dist_device_sync` will be used which uses a parameter server to manage multiple hosts and performs the gradient updates on the GPUs when possible.
# 2. Shard the training dataset.
#   * To perform distributed cluster training, the training dataset is split into shards with at least 1 shard per host.
#   * In this case it is split into 5 shards using 5 hosts.
#   * Each host trains using only a portion of the dataset.
#   * The sharded training data is stored in Amazon S3.
# 3. Split each batch into portions and copy the portions onto one GPU per portion.
#   * In this case 4 GPUs will be used.
#   * Each GPU trains on only a portion of each batch.
#   * The gradients are summed over all GPUs at the end of the batch and all GPUs (and hosts when combining with distributed) are updated.
#   * These updates are performed on the GPU when possible.
#   * You will perform the splitting, and Gluon will automatically manage the synchronizing and updates.
# 
# ### Split Training and Test, Shard the Training Data, and Upload to S3
# 
# The first 80% of the time series is used for training and the last 20% is used as a test set.
# 
# The training set is sharded sequentially into 2 parts, one for each host in the cluster.
# 
# These datasets are written to a CSV file and then uploaded to Amazon S3 to be used in training.
# <div style="text-align: right"><a href="#toc">Back to top</a></div>

#%%
single_host_train_bucket_prefix = '/train/single_host/'
multiple_host_train_bucket_prefix = '/train/multiple_host/'

splits = 2
train_frac = 0.8

num_time_steps = len(df)
split_index = int(num_time_steps*train_frac)
train = df[0:split_index]
print('Training size {}'.format(len(train)))
test = df[split_index:]
print('Test size {}'.format(len(test)))

train_sets = []
train_len = len(train)
train_size = int(train_len)/splits
for i in range(0,splits):
    start = int(i*train_size)
    end = int((i+1)*train_size)
    print('start {}'.format(start))
    print('end {}'.format(end))
    if end < (train_len-1):
        train_sets.append(train[start:end])
    else:
        train_sets.append(train[start:])

test_file_path = os.path.join(data_dir,'test.csv')
test.to_csv(test_file_path,header=None,index=False)
train_file_path = os.path.join(data_dir,'train.csv')
train.to_csv(train_file_path,header=None,index=False)

client = boto3.client('s3')

for i in range(0,splits):
    file_path = os.path.join(data_dir,'train_{}.csv'.format(i))
    print('Uploading file: {} with {} rows'.format(file_path,len(train_sets[i])))
    train_sets[i].to_csv(file_path,header=None,index=False)
    s3_path = prefix + '{}train_{}.csv'.format(multiple_host_train_bucket_prefix,i)
    print('Uploading to {}'.format(s3_path))
    client.upload_file(file_path, bucket, s3_path)

client.upload_file(test_file_path, bucket, prefix + '{}test.csv'.format(test_bucket_prefix))
client.upload_file(train_file_path, bucket, prefix + '{}train.csv'.format(single_host_train_bucket_prefix))

#%% [markdown]
# ### Modifications to `lstnet_sagemaker.py` when using multiple GPUs/hosts
# 
# There are two main changes to the module:
#     
# 1. Set the `kvstore` to `dist_device_sync` when multiple GPUs and hosts are available.
# 1. Split each batch into one part per GPU and copy each part to a separate GPU before training.

#%%
get_ipython().system('cat lstnet_sagemaker.py')

#%% [markdown]
# ### Test Locally
# 
# To make sure there are no obvious bugs in the code, call the `train()` function. This is done with 1 epoch to verify that it executed correctly. There are also some basic unit tests included in the module.
# 
# The key parameters to the `train()` function in this case are:
# 
# * `hyperparameters`: The Amazon SageMaker Hyperparameters dictionary. A dictionary of string-to-string.
# * `channel_input_dirs`: A dictionary of string-to-string maps from the Amazon SageMaker algorithm input channel name to the directory containing files for that input channel.
#   * **Note:** If the Amazon SageMaker training job is run in PIPE mode, this dictionary will be empty.
# * `output_data_dir`: The Amazon SageMaker output data directory. After the function returns, data written to this directory is made available in the Amazon SageMaker training job output location.
# * `num_gpus`: The number of GPU devices available on the host this script is being executed on. As we are running on a CPU notebook instance, this number is set to zero.
# * `num_cpus`: The number of CPU devices available on the host this script is being executed on.
# * `hosts`: A list of hostnames in the Amazon SageMaker training job cluster.
# * `current_host`: This host's name. It will exist in the hosts list.
# * `kwargs`: Other keyword agruments.

#%%
from lstnet_sagemaker import train
hyperparameters = {
    'conv_hid' : 10,
    'gru_hid' : 10,
    'skip_gru_hid' : 2,
    'skip' : 5,
    'ar_window' : 6,
    'window' : 24*7,
    'horizon' : 24,
    'learning_rate' : 0.01,
    'clip_gradient' : 10.,
    'batch_size' : 128,
    'epochs' : 1
}
channel_input_dirs = {'train':data_dir,'test':data_dir}
train(
    hyperparameters = hyperparameters,
    input_data_config = None,
    channel_input_dirs = channel_input_dirs,
    output_data_dir = os.path.join(data_dir,'output'),
    model_dir = None,
    num_gpus = 0,
    num_cpus = 1,
    hosts = ['localhost'],
    current_host = 'localhost')

#%% [markdown]
# ### Choose Hyperparameters
# 
# One parameter to watch when switching to multi-GPU is `batch_size`. When using multiple GPUs, each batch is split across all available GPUs. To improve performance, it is common to increase the batch size. Increasing the number of epochs may increase accuracy but will also increase training time.

#%%
hyperparameters = {
    'conv_hid' : 100,
    'gru_hid' : 100,
    'skip_gru_hid' : 5,
    'skip' : 24,
    'ar_window' : 24,
    'window' : 24*7,
    'horizon' : 24,
    'learning_rate' : 0.0001,
    'clip_gradient' : 10.,
    'batch_size' : 512,
    'epochs' : 10
}

#%% [markdown]
# ### Trigger the Training Job Using the Amazon SageMaker Python SDK
# 
# The main differences to moving from a single host and GPU, include:
#   * The `train_instance_count` is set to 2 to run on a 2-node CPU cluster.
#   * The `train_instance_type` of `ml.p3.2xlarge` is chosen with a single GPU. The code is designed so that you could take this and run it on multi-GPU instances by only changing the instance type.
# 
# Amazon SageMaker automatically bootstraps an MXNet cluster with 2 nodes.
# 
# First an estimator is created with `sagemaker.mxnet.MXNet`. The inputs are:
#   * `entry_point='lstnet_sagemaker.py'`: The module used to run the training by calling the `train()` function.
#   * `source_dir='.'`: An optional directory containing code is copied onto the Amazon SageMaker training hosts and made available to the training script.
#   * `role=role`: The IAM role which is given to the training hosts giving them privileges such as access to the Amazon S3 bucket.
#   * `output_path='s3://{}/{}/output'.format(bucket, prefix)`: The Amazon S3 bucket to store artifacts such as the model parameters.
#   * `train_instance_count=2`: The number of hosts used for training.
#     * Using a number > 1 will start a cluster.
#     * To take advantage of this, the training data is sharded.
#   * `train_instance_type='ml.p3.2xlarge'`: The Amazon EC2 instance type to be used for training hosts.
#     * In this case, choose the latest generation `p3` instance type, each of which contains 1 Nvidia Tesla v100 GPU.
#   * `hyperparameters=hyperparameters`: The hyperparameter dictionary made available to the `train()` function in the endpoint script.
# 
# Then, the `fit()` method of the estimator is called. The parameters are:
# 
#   * `inputs`: A dictionary containing the URLs in Amazon S3 of the `train/` data directory and the `test/` data directory.
#   * `wait`: This is specified as `False` so the `fit()` method returns immediately after the training job is created.
#     * Go to the Amazon SageMaker console to monitor the progress of the job.
#     * Set `wait` to `True` to block and see the progress of the training job output in the notebook. If you leave it as `False` the training jobs will run in the background and allow you to continue with the notebook.
# 
# You will run two different versions to compare training speeds:
# 
# 1. Two hosts each with 4 CPUs (`ml.m4.xlarge`)
# 1. One host with 1 GPU (`ml.p3.2xlarge`)
# 
# **If you receive an error with 'resource limit exceeded' try reducing the number of instances or changing the instance type from a GPU to a CPU instance**
# 
# **Challenge:** Experiment with more combinations to improve performance. Can you find the most efficient batch-size vs hardware combination for this network? Which job is faster - a single GPU or multiple CPUs? Why?

#%%
lstnet1 = MXNet(entry_point='lstnet_sagemaker.py',
                source_dir='.',
                role=role,
                output_path='s3://{}/{}/output'.format(bucket, prefix),
                train_instance_count=2,
                train_instance_type='ml.m4.xlarge',
                hyperparameters=hyperparameters,
                framework_version=1.2)
lstnet1.fit(inputs={
                'train': 's3://{}/{}{}'.format(bucket, prefix, multiple_host_train_bucket_prefix),
                'test': 's3://{}/{}{}'.format(bucket, prefix, test_bucket_prefix)
            },
            wait=False)


#%%
lstnet3 = MXNet(entry_point='lstnet_sagemaker.py',
    source_dir='.',
    role=role,
    output_path='s3://{}/{}/output'.format(bucket, prefix),
    train_instance_count=1,
    train_instance_type='ml.p3.2xlarge',
    hyperparameters=hyperparameters, 
    framework_version=1.2)
lstnet3.fit(inputs={
                'train': 's3://{}/{}{}'.format(bucket, prefix, single_host_train_bucket_prefix),
                'test': 's3://{}/{}{}'.format(bucket, prefix, test_bucket_prefix)
            },
            wait=False)

#%% [markdown]
# ___
# 
# 
# ## Section 3: Challenge - Deploying an Endpoint Using Amazon SageMaker
# <a name="section3"></a>
# 
# Amazon SageMaker is composed of 5 main services:
#   * Hosted Notebooks
#   * Built-in Algorithms
#   * Model Training
#   * Hyperparameter Tuning
#   * Model Hosting
# 
# Each service works together, and can also be used independendly as needed. In this section, you will learn how to deploy and host a model so it can be used in production to evaluate novel inputs. The hosted endpoints can either be used on single examples or in batch mode over a larger number of examples.
# 
# In this challenge portion of the lab you have the option to explore three different ways of deploying the endpoint. You can either choose the option that you feel most comfortable with or challenge yourself with an option you may be less familiar with. If there is time remaining, feel free to work through all three options.
# 
# 1. <a href="#s3p1">Using a model trained with Amazon SageMaker. (Easy)</a>  
# 1. <a href="#s3p2">From a model trained elsewhere with artifacts stored in Amazon S3. (Moderate)</a> 
# 1. <a href="#s3p3">From a model developed with a custom Docker container. (Harder)</a>
# <div style="text-align: right"><a href="#toc">Back to top</a></div>

#%%
sagemaker_session = sagemaker.Session()

#%% [markdown]
# <a name="s3p1"></a>
# ### Challenge Option 1: Deploying a Model Trained with Amazon SageMaker
# 
# 
# A model is trained using Amazon SageMaker and then directly deployed to an endpoint.
# 
# MNIST is a widely used dataset for handwritten digit classification. It consists of 70,000 labeled 28x28 pixel grayscale images of hand-written digits. The dataset is split into 60,000 training images and 10,000 test images. There are 10 classes (one for each of the 10 digits). In this part of the lab, you will train and test an MNIST model on Amazon SageMaker using MXNet and the Gluon API.
# 
# #### Download Training and Test Data

#%%
gluon.data.vision.MNIST('./data/train', train=True)
gluon.data.vision.MNIST('./data/test', train=False)

#%% [markdown]
# #### Uploading the Data
# 
# Use the `sagemaker.Session.upload_data` function to upload your datasets to an Amazon S3 location. The return value `inputs` identifies the location. You will use this later when starting the training job.
# 
# **Note:** Save the bucket name that is included in the output of the below code cell. It will look similar to `sagemaker-REGION-ACCOUNT_ID`.

#%%
inputs = sagemaker_session.upload_data(path='data', key_prefix='data/DEMO-mnist')
inputs

#%% [markdown]
# #### Implement the Training Function
# 
# You will need to provide a training script that can run on the Amazon SageMaker platform. The training scripts are the same as the ones you would write for local training, except you need to provide a `train` function. When Amazon SageMaker calls your function, it will pass in arguments that describe the training environment. Check the script below to see how this works.
# 
# The script here is an adaptation of the [Gluon MNIST example](https://github.com/gluon-api/gluon-api/blob/master/tutorials/mnist-gluon-example.ipynb) provided by the [Apache MXNet](https://mxnet.incubator.apache.org/) project. 

#%%
get_ipython().system('cat mnist.py')

#%% [markdown]
# #### Run the Training Script on Amazon SageMaker
# 
# The `MXNet` class allows you to run your training function on Amazon SageMaker infrastructure. You need to configure it with your training script, an IAM role, the number of training instances, and the training instance type. In this case, you will run your training job on a single `c4.xlarge` instance.
# 
# After you have constructed your `MXNet` object, fit it using the data uploaded to Amazon S3. Amazon SageMaker makes sure your data is available in the local filesystem, so your training script can simply read the data from disk.
# 
# **Note:** You can ignore any warnings outputted by the below code cell. 

#%%
m = MXNet("mnist.py",
          role=role,
          train_instance_count=1,
          train_instance_type="ml.c4.xlarge",
          hyperparameters={'batch_size': 100,
                         'epochs': 10,
                         'learning_rate': 0.1,
                         'momentum': 0.9,
                         'log_interval': 100},
          framework_version=1.2)
m.fit(inputs)

#%% [markdown]
# After training, use the `MXNet` object to build and deploy an `MXNetPredictor` object. This creates an Amazon SageMaker endpoint that you can use to perform inference on JSON-encoded multidimensional arrays.
# 
# The `deploy` method does the following, in order:
# 
# 1. Creates an Amazon SageMaker model by calling the `CreateModel` API. The model that you create in Amazon SageMaker holds information such as location of the model artifacts and the inference code image.
# 1. Creates an endpoint configuration by calling the `CreateEndpointConfig` API. This configuration holds necessary information including the name of the model (which was created in the preceding step) and the resource configuration (the type and number of Machine Learning compute instances to launch for hosting).
# 1. Creates the endpoint by calling the `CreateEndpoint` API and specifying the endpoint configuration created in the preceding step. Amazon SageMaker launches Machine Learning compute instances as specified in the endpoint configuration, and deploys the model on them.
# 
# Launching an endpoint can take up to 10 minutes to complete

#%%
predictor = m.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

#%% [markdown]
# You can now use this predictor to classify hand-written digits. Drawing into the image box loads the pixel data into a `data` variable in this notebook, which can then be passed to the MXNet predictor. 

#%%
HTML(open("input.html").read())

#%% [markdown]
# The predictor runs inference on your input data and returns the predicted digit.

#%%
response = predictor.predict(data)
print(int(response))

#%% [markdown]
# #### Cleanup
# 
# After you have finished with this example, remember to delete the prediction endpoint to release the instance(s) associated with it.

#%%
sagemaker.Session().delete_endpoint(predictor.endpoint)

#%% [markdown]
# <a name="s3p2"></a>
# ### Challenge Option 2: Deploying from Amazon S3
# 
# 
# Amazon SageMaker saves and stores the artifacts in Amazon S3. In the case where you have already trained a model elsewhere, it is relatively straightforward to upload the model artifacts to Amazon S3 and deploy the model using Amazon SageMaker.
# 
# In this case, use the model trained in the previous section. You will create a model using the [sagemaker.mxnet.model.MXNetModel](http://sagemaker.readthedocs.io/en/latest/sagemaker.mxnet.html#mxnet-model) function, passing the location in Amazon S3 with the zipped and archived copy of the model artifacts, and then deploy it using Amazon S3.
# 
# #### Locate the model artifacts in S3
# 
# The model in the previous section saved the artifacts including the parameters and model declaration in Amazon S3. This will be in the default bucket used by the notebook (`sagemaker-REGION-ACCOUNT_ID`). Locate this bucket and scroll down to find the latest folder that was added. It will be named the same as the endpoint that was deleted in the code cell above. Inside this folder, under `output`, you will find a `model.tar.gz` file. Download this, unpack it, and have a look at the contents. Here is an example path:
# 
# `BUCKET_NAME/sagemaker-mxnet-2018-07-20-08-45-48-381/output/model.tar.gz`
# 
# Two files are contained:
# 
# * `model.json`: The declaration of the network.
# * `model.params`: The trained parameters of the model.
# 
# #### Copy the model artifacts to a new location
# 
# Copy the `model.tar.gz` into your lab bucket.
# 
# #### Load the model using MXNetModel
# 
# Load model from S3 using [sagemaker.mxnet.model.MXNetModel](http://sagemaker.readthedocs.io/en/latest/sagemaker.mxnet.html#mxnet-model).

#%%
model_data = "s3://%s/model.tar.gz" % bucket

m2 = MXNetModel(model_data, role, "mnist.py")

predictor = m2.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

#%% [markdown]
# This **may** fail. If so, it is because you already have a model set up with the same configuration. How do you think it knows that it is the same model? Go to the Amazon SageMaker console and find the model under Inference -> Models and copy the name of the model. Replace **MODEL_NAME** in the code cell below with the name of your model.

#%%
########################################################
# Only run this cell if the previous code cell failed! #
########################################################
model_name = 'MODEL_NAME'

sagemaker_client = boto3.client('sagemaker')
sagemaker_client.delete_model(ModelName=model_name)
predictor = m2.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

#%% [markdown]
# Use the same input you drew in the cell above and score against this new endpoint. If you would like to do so, feel free to clear the input box and try additional numbers.

#%%
response = predictor.predict(data)
print(int(response))

#%% [markdown]
# #### Cleanup
# 
# After you have finished with this example, remember to delete the prediction endpoint to release the instance(s) associated with it.

#%%
sagemaker.Session().delete_endpoint(predictor.endpoint)

#%% [markdown]
# <a name="s3p3"></a>
# ### Challenge Option 3: Deploying Using a Custom Docker Container
# 
# 
# With Amazon SageMaker, you can package your own algorithms that can then be trained and deployed in the Amazon SageMaker environment. This notebook will guide you through an example that shows you how to build a Docker container for Amazon SageMaker and use it for training and inference.
# 
# By packaging an algorithm in a container, you can bring almost any code to the Amazon SageMaker environment, regardless of programming language, environment, framework, or dependencies.
# 
# 1. [Building your own algorithm container](#Building-your-own-algorithm-container)
#   1. [When should I build my own algorithm container?](#When-should-I-build-my-own-algorithm-container?)
#   1. [The example](#The-example)
# 1. [Packaging and Uploading your Algorithm for use with Amazon SageMaker](#Part-1:-Packaging-and-Uploading-your-Algorithm-for-use-with-Amazon-SageMaker)
#     1. [An overview of Docker](#An-overview-of-Docker)
#     1. [How Amazon SageMaker runs your Docker container](#How-Amazon-SageMaker-runs-your-Docker-container)
#       1. [Running your container during training](#Running-your-container-during-training)
#       1. [Running your container during hosting](#Running-your-container-during-hosting)
#     1. [The Dockerfile](#The-Dockerfile)
#     1. [Building and registering the container](#Building-and-registering-the-container)
#   1. [Testing your algorithm on your local machine or on an Amazon SageMaker notebook instance](#Testing-your-algorithm-on-your-local-machine-or-on-an-Amazon-SageMaker-notebook-instance)
# 1. [Part 2: Training and Hosting your Algorithm in Amazon SageMaker](#Part-2:-Training-and-Hosting-your-Algorithm-in-Amazon-SageMaker)
#   1. [Set up the environment](#Set-up-the-environment)
#   1. [Create the session](#Create-the-session)
#   1. [Upload the data for training](#Upload-the-data-for-training)
#   1. [Create an estimator and fit the model](#Create-an-estimator-and-fit-the-model)
#   1. [Deploy the model](#Deploy-the-model)
#   1. [Choose some data and use it for a prediction](#Choose-some-data-and-use-it-for-a-prediction)
#   1. [Optional cleanup](#Optional-cleanup)  
# 
# _or_ if you're feeling a little impatient, you can jump directly [to the code](#The-Dockerfile)!
# 
# #### Building your Own Algorithm Container
# <a name="Building-your-own-algorithm-container"></a>
# 
# ##### When Should I Build my Own Algorithm Container?
# <a name="When-should-I-build-my-own-algorithm-container?"></a>
# 
# You may not need to create a container to bring your own code to Amazon SageMaker. When you are using a framework (such as Apache MXNet or TensorFlow) that has direct support in Amazon SageMaker, you can simply supply the Python code that implements your algorithm using the SDK entry points for that framework. This set of frameworks is continually expanding, so we recommend that you check the current list if your algorithm is written in a common machine learning environment.
# 
# Even if there is direct SDK support for your environment or framework, you may find it more effective to build your own container. If the code that implements your algorithm is quite complex on its own or you need special additions to the framework, building your own container may be the right choice.
# 
# If there isn't direct SDK support for your environment, don't worry. You'll see in this walk-through that building your own container is quite straightforward.
# 
# ##### The Example
# <a name="The-example"></a>
# 
# Here, you'll see how to package a simple Python example which showcases the decision tree algorithm from the widely used `scikit-learn` machine learning package. The example is purposefully fairly trivial since the point is to show the surrounding structure that you'll want to add to your own code so you can train and host it in Amazon SageMaker.
# 
# The ideas shown here will work in any language or environment. You'll need to choose the right tools for your environment to serve HTTP requests for inference, but good HTTP environments are available in every language these days.
# 
# In this example, you will use a single image to support training and hosting. This is easy because it means that you only need to manage one image and can set it up to do everything. Sometimes you'll want separate images for training and hosting because they have different requirements. Just separate the parts discussed below into separate Dockerfiles and build two images. Choosing whether to have a single image or two images is really a matter of which is more convenient for you to develop and manage.
# 
# If you're only using Amazon SageMaker for training or hosting, but not both, there is no need to build the unused functionality into your container.
# 
# [scikit-learn]: http://scikit-learn.org/stable/
# [decision tree]: http://scikit-learn.org/stable/modules/tree.html
# 
# #### Packaging and Uploading your Algorithm for use with Amazon SageMaker
# <a name="Part-1:-Packaging-and-Uploading-your-Algorithm-for-use-with-Amazon-SageMaker"></a>
# 
# ##### An Overview of Docker
# <a name="An-overview-of-Docker"></a>
# 
# If you're familiar with Docker already, you can skip ahead to the next section.
# 
# For many data scientists, Docker containers are a new concept, but they are not difficult, as you'll see here.
# 
# Docker provides a simple way to package arbitrary code into an _image_ that is totally self-contained. Once you have an image, you can use Docker to run a _container_ based on that image. Running a container is just like running a program on the machine except that the container creates a fully self-contained environment for the program to run. Containers are isolated from each other and from the host environment, so the way you set up your program is the way it runs, no matter where you run it.
# 
# Docker is more powerful than environment managers like conda or virtualenv because (a) it is completely language independent and (b) it comprises your whole operating environment, including startup commands, environment variable, etc.
# 
# In some ways, a Docker container is like a virtual machine, but it is much lighter weight. For example, a program running in a container can start in less than a second and many containers can run on the same physical machine or virtual machine instance.
# 
# Docker uses a simple file called a `Dockerfile` to specify how the image is assembled. We'll see an example of that below. You can build your Docker images based on Docker images built by yourself or others, which can simplify things quite a bit.
# 
# Docker has become very popular in the programming and devops communities for its flexibility and well-defined specification of the code to be run. It is the underpinning of many services built in the past few years, such as Amazon Elastic Container Service (ECS).
# 
# Amazon SageMaker uses Docker to allow users to train and deploy arbitrary algorithms.
# 
# In Amazon SageMaker, Docker containers are invoked in a certain way for training and a slightly different way for hosting. The following sections outline how to build containers for the Amazon SageMaker environment.
# 
# Some helpful links:
# 
# * [Docker home page](http://www.docker.com)
# * [Getting started with Docker](https://docs.docker.com/get-started/)
# * [Dockerfile reference](https://docs.docker.com/engine/reference/builder/)
# * [`docker run` reference](https://docs.docker.com/engine/reference/run/)
# 
# [Amazon ECS]: https://aws.amazon.com/ecs/
# 
# ##### How Amazon SageMaker Runs your Docker Container
# <a name="How-Amazon-SageMaker-runs-your-Docker-container"></a>
# 
# Because you can run the same image in training or hosting, Amazon SageMaker runs your container with the argument `train` or `serve`. How your container processes this argument depends on the container:
# 
# * In the example here, an `ENTRYPOINT` is not defined in the Dockerfile so Docker will run the command `train` at training time and `serve` at serving time. In this example, these are defined as executable Python scripts, but they could be any program that you want to start in that environment.
# * If you specify a program as an `ENTRYPOINT` in the Dockerfile, that program will be run at startup and its first argument will be `train` or `serve`. The program can then look at that argument and decide what to do.
# * If you are building separate containers for training and hosting (or building only for one or the other), you can define a program as an `ENTRYPOINT` in the Dockerfile and ignore (or verify) the first argument passed in.
# 
# ###### Running your Container During Training
# <a name="Running-your-container-during-training"></a>
# 
# When Amazon SageMaker runs training, your `train` script is run just like a regular Python program. A number of files are laid out for your use, under the `/opt/ml` directory:
# 
#     ```plain
#     /opt/ml
#     ├── input
#     │   ├── config
#     │   │   ├── hyperparameters.json
#     │   │   └── resourceConfig.json
#     │   └── data
#     │       └── <channel_name>
#     │           └── <input data>
#     ├── model
#     │   └── <model files>
#     └── output
#         └── failure
#     ```
# 
# **The Input**
# 
# * `/opt/ml/input/config` contains information to control how your program runs.
#   * `hyperparameters.json` is a JSON-formatted dictionary of hyperparameter names to values. These values will always be strings, so you may need to convert them.
#   * `resourceConfig.json` is a JSON-formatted file that describes the network layout used for distributed training. Since scikit-learn doesn't support distributed training, ignore it here.
# * `/opt/ml/input/data/<channel_name>/` (for File mode) contains the input data for that channel. The channels are created based on the call to `CreateTrainingJob` but it's generally important that channels match what the algorithm expects. The files for each channel will be copied from Amazon S3 to this directory, preserving the tree structure indicated by the S3 key structure.
# * `/opt/ml/input/data/<channel_name>_<epoch_number>` (for Pipe mode) is the pipe for a given epoch. Epochs start at zero and go up by one each time you read them. There is no limit to the number of epochs that you can run, but you must close each pipe before reading the next epoch.
# 
# **The Output**
# 
# * `/opt/ml/model/` is the directory where you write the model that your algorithm generates. Your model can be in any format that you want. It can be a single file or a whole directory tree. Amazon SageMaker will package any files in this directory into a compressed tar archive file. This file will be available at the S3 location returned in the `DescribeTrainingJob` result.
# * `/opt/ml/output` is a directory where the algorithm can write a file `failure` that describes why the job failed. The contents of this file will be returned in the `FailureReason` field of the `DescribeTrainingJob` result. For jobs that succeed, there is no reason to write this file as it will be ignored.
# 
# ###### Running your Container During Hosting
# <a name="Running-your-container-during-hosting"></a>
# 
# Hosting has a very different model than training because hosting is reponding to inference requests that come in via HTTP. In this example, you will use a Python serving stack to provide robust and scalable serving of inference requests:
# 
# ![](./stack.png)
# 
# This stack is implemented in the sample code here and you can mostly just leave it alone.
# 
# Amazon SageMaker uses two URLs in the container:
# 
# * `/ping` will receive `GET` requests from the infrastructure. Your program returns `200` if the container is up and accepting requests.
# * `/invocations` is the endpoint that receives client inference `POST` requests. The format of the request and the response is up to the algorithm. If the client supplied `ContentType` and `Accept` headers, these will be passed in as well.
# 
# The container will have the model files in the same place they were written during training:
# 
#     ```plain
#     /opt/ml
#     └── model
#         └── <model files>
#     ```
# 
# ##### The Dockerfile
# <a name="The-Dockerfile"></a>
# 
# The Dockerfile describes the image that you want to build. You can think of it as describing the complete operating system installation of the system that you want to run. A Docker container running is quite a bit lighter than a full operating system, however, because it takes advantage of Linux on the host machine for the basic operations.
# 
# For the Python science stack, start from a standard Ubuntu installation and run the normal tools to install the things needed by `scikit-learn`. Finally, add the code that implements your specific algorithm to the container and set up the right environment to run under.
# 
# Along the way, clean up extra space. This makes the container smaller and faster to start.
# 
# Look at the Dockerfile for the example:

#%%
get_ipython().system('cat container/Dockerfile')

#%% [markdown]
# ##### Building and Registering the Container
# <a name="Building-and-registering-the-container"></a>
# 
# The following shell code shows how to build the container image using `docker build` and push the container image to Amazon ECR using `docker push`.
# 
# This code looks for an Amazon ECR repository in the account you're using and the current default region (if you're using a Amazon SageMaker notebook instance, this will be the region where the notebook instance was created). If the repository doesn't exist, the script will create it.

#%%
get_ipython().run_cell_magic('sh', '', '\n# The name of our algorithm\nalgorithm_name=decision-trees-sample\n\ncd container\n\nchmod +x decision_trees/train\nchmod +x decision_trees/serve\n\naccount=$(aws sts get-caller-identity --query Account --output text)\n\n# Get the region defined in the current configuration (default to us-west-2 if none defined)\nregion=$(aws configure get region)\nregion=${region:-us-west-2}\n\nfullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"\n\n# If the repository doesn\'t exist in ECR, create it.\n\naws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1\n\nif [ $? -ne 0 ]\nthen\n    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null\nfi\n\n# Get the login command from ECR and execute it directly\n$(aws ecr get-login --region ${region} --no-include-email)\n\n# Build the docker image locally with the image name and then push it to ECR\n# with the full name.\n\ndocker build  -t ${algorithm_name} .\ndocker tag ${algorithm_name} ${fullname}\n\ndocker push ${fullname}')

#%% [markdown]
# ###### Testing your Algorithm on your Local Machine or on an Amazon SageMaker Notebook Instance
# <a name="Testing-your-algorithm-on-your-local-machine-or-on-an-Amazon-SageMaker-notebook-instance"></a>
# 
# While you're first packaging an algorithm to use with Amazon SageMaker, you probably want to test it yourself to make sure it's working right. In the directory `local_test`, there is a framework for doing this. It includes three shell scripts for running and using the container and a directory structure that mimics the one outlined above.
# 
# The scripts are:
# 
# * `train_local.sh`: Run this with the name of the image and it will run training on the local tree. You'll want to modify the directory `test_dir/input/data/...` to be set up with the correct channels and data for your algorithm. Also, you'll want to modify the file `input/config/hyperparameters.json` to have the hyperparameter settings that you want to test (as strings).
# * `serve_local.sh`: Run this with the name of the image once you've trained the model and it should serve the model. It will run and wait for requests. Simply use the keyboard interrupt to stop it.
# * `predict.sh`: Run this with the name of a payload file and (optionally) the HTTP content type you want. The content type will default to `text/csv`. For example, you can run `$ ./predict.sh payload.csv text/csv`. To test predict pass a csv file with 4 columns of floats. A single row will work.
# 
# The directories as shipped are set up to test the decision tree's sample algorithm presented here.
# 
# #### Training and Hosting your Algorithm in Amazon SageMaker
# <a name="Part-2:-Training-and-Hosting-your-Algorithm-in-Amazon-SageMaker"></a>
# 
# Once you have your container packaged, you can use it to train and serve models. Do that with the algorithm you made above.
# 
# ##### Set up the Environment
# <a name="Set-up-the-environment"></a>
# 
# Here you specify a bucket prefix to use and the role that will be used for working with Amazon SageMaker.

#%%
# S3 prefix
prefix = 'DEMO-scikit-byo-iris'

import boto3
import re
import itertools
import os
import numpy as np
import pandas as pd
from sagemaker import get_execution_role
from time import gmtime, strftime

role = get_execution_role()

#%% [markdown]
# ##### Create the Session
# <a name="Create-the-session"></a>
# 
# The session remembers your connection parameters to Amazon SageMaker. You will use it to perform all of your Amazon SageMaker operations.

#%%
session = sagemaker.Session()

#%% [markdown]
# ##### Upload the Data for Training
# <a name="Upload-the-data-for-training"></a>
# 
# When training large models with huge amounts of data, you'll typically use big data tools, like Amazon Athena, AWS Glue, or Amazon EMR, to create your data in S3. For the purposes of this example, you're using some of the classic [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set), which has been included.
# 
# We can use the tools provided by the Amazon SageMaker Python SDK to upload the data to a default bucket. 

#%%
WORK_DIRECTORY = 'iris_data'
data_location = session.upload_data(WORK_DIRECTORY, key_prefix=prefix)
print(data_location)

#%% [markdown]
# ##### Create an Estimator and Fit the Model
# <a name="Create-an-estimator-and-fit-the-model"></a>
# 
# In order to use Amazon SageMaker to fit your algorithm, you'll create an `Estimator` that defines how to use the container to train. This includes the configuration needed to invoke Amazon SageMaker training:
# 
# * The __container name__. This is constructed as in the shell commands above.
# * The __role__. As defined above.
# * The __instance count__ which is the number of machines to use for training.
# * The __instance type__ which is the type of machine to use for training.
# * The __output path__ determines where the model artifact will be written.
# * The __session__ is the Amazon SageMaker session object that you defined above.
# 
# Then you'll use `fit()` on the estimator to train against the data you uploaded previously.

#%%
account = session.boto_session.client('sts').get_caller_identity()['Account']
region = session.boto_session.region_name
image = '{}.dkr.ecr.{}.amazonaws.com/decision-trees-sample:latest'.format(account, region)

tree = sagemaker.estimator.Estimator(image,
                                     role,
                                     1,
                                     'ml.c4.2xlarge',
                                     output_path="s3://{}/output".format(session.default_bucket()),
                                     sagemaker_session=session)

tree.fit(data_location)

#%% [markdown]
# ##### Deploy the Model
# <a name="Deploy-the-model"></a>
# 
# Deploying the model to Amazon SageMaker hosting just requires a `deploy` call on the fitted model. This call takes an instance count, instance type, and optionally serializer and deserializer functions. These are used when the resulting predictor is created on the endpoint.

#%%
from sagemaker.predictor import csv_serializer
predictor = tree.deploy(1, 'ml.m4.xlarge', serializer=csv_serializer)

#%% [markdown]
# ##### Choose Some Data and Use it for a Prediction
# <a name="Choose-some-data-and-use-it-for-a-prediction"></a>
# 
# In order to do some predictions, extract some of the data used for training and do predictions against it. This is, of course, bad statistical practice, but a good way to see how the mechanism works.

#%%
shape=pd.read_csv("iris_data/iris.csv",header=None)

a = [50*i for i in range(3)]
b = [40+i for i in range(10)]
indices = [i+j for i,j in itertools.product(a,b)]

test_data=shape.iloc[indices[:-1]]
test_X=test_data.iloc[:,1:]
test_y=test_data.iloc[:,0]

#%% [markdown]
# Prediction is as easy as calling predict with the predictor returned from the `deploy` command and the data you want to do predictions with. The serializers take care of doing the data conversions for you.

#%%
print(predictor.predict(test_X.values).decode('utf-8'))

#%% [markdown]
# ##### Optional Cleanup
# <a name="Optional-cleanup"></a>
# 
# When you're done with the endpoint, you'll want to clean it up.

#%%
session.delete_endpoint(predictor.endpoint)

#%% [markdown]
# ## Break: Instructor Review
# 
# ___
# 
# Now that you have completed the last challenge-focused aspect of this lab, take a pause and listen to the instructor review some of the key steps you just performed. This will help ensure you fully understand the key aspects of what has been covered in this lab.
#%% [markdown]
# ## Lab Complete
# 
# Congratulations! You have completed this lab. To clean up your lab environment, do the following:
# 
# 1. To sign out of the AWS Management Console, click **awsstudent** at the top of the console, and then click **Sign Out**.
# 1. On the Qwiklabs page, click **End**.

#%%



