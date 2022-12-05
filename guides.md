# Step0: data preparation(optional) on  System terminal of SageMaker Studio notebook instance

* this step should be done on EC2 (Choose Deep Learning AMI ubuntu， g4dn.2xlarge + 250G EBS) for convenience, script may need to be modified as needed.
```
#configure independent environment
cd ~
mkdir transsionpoc && cd transsionpoc
git clone https://github.com/wenet-e2e/wenet.git
conda create -n wenet python=3.8
conda activate wenet
cd wenet
pip install -r requirements.txt
conda install pytorch=1.10.0 torchvision torchaudio=0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

```
# prepare dataset
cd examples/librispeech/s0/
chmod 777 *.sh
#mkdir export
sudo mkdir -p /opt/ml/input/training
sed -i "s/\/export\/data\/en-asr-data\/OpenSLR/\/opt\/ml\/input\/training/g" run.sh

# install flac
# run "wget https://downloads.xiph.org/releases/flac/flac-1.3.2.tar.xz" on your pc to download this file or use provided one.
# then upload to ~/transsionpoc/wene/texamples/librispeech/s0 folder
xz -d flac-1.3.2.tar.xz && tar -xvf flac-1.3.2.tar && cd flac-1.3.2 && ./configure --prefix=/usr --disable-thorough-tests && make && make install

# modify run.sh to download dataset files, stage -1: Data Download: 
cd ..
sed -i "s/stage=0/stage=-1/g" run.sh
sed -i "s/stop_stage=5/stop_stage=-1/g" run.sh
sudo ./run.sh

# modify run.sh to prepare data, stage 0: Data preparation: 
sed -i "s/stage=-1/stage=0/g" run.sh
sed -i "s/stop_stage=-1/stop_stage=0/g" run.sh
 ./run.sh

# modify run.sh to generate features, stage 1: Feature Generation: 
chmod 777 data/train_960
sudo chmod 777 data/train_960/text
sudo chmod 777 data/train_960/wav.scp
sudo chmod 777 data/dev/text
sudo chmod 777 data/dev/wav.scp
sudo chmod 777 data/train_960

sed -i "s/stage=0/stage=1/g" run.sh
sed -i "s/stop_stage=0/stop_stage=1/g" run.sh
sudo ./run.sh

# modify run.sh to prepare dictionary, stage 2: Dictionary and Json Data Preparatio: 
sudo chmod 777 data/lang_char
sudo chmod 777 data/lang_char/train_960_unigram5000_units.txt
sudo chmod 777 data/lang_char/input.txt

sed -i "s/stage=1/stage=2/g" run.sh
sed -i "s/stop_stage=1/stop_stage=2/g" run.sh
 ./run.sh

# modify run.sh to Prepare data, prepare required format , stage 3: Prepare data, prepare required format: 

sed -i "s/stage=2/stage=3/g" run.sh
sed -i "s/stop_stage=2/stop_stage=3/g" run.sh
sudo ./run.sh

# modify run.sh to start training, stage 4: Training: 
#sed -i "s/stage=3/stage=4/g" run.sh
#sed -i "s/stop_stage=3/stop_stage=4/g" run.sh
#sudo ./run.sh
```

# Step1: upload training dataset to s3 bucket

<bucket>/<prefix> must be the same used in the jupyter notebook file.
```
#upload dataset+etc to s3 bucket
cd ~/transsionpoc/wenet/examples/librispeech/s0/
aws s3 cp --recursive  exp s3://<bucket>/<prefix>/exp
aws s3 cp --recursive  data s3://<bucket>/<prefix>/data
#make sure directory data/train_960 exist
sudo rm -fr /opt/ml/input/training/*.tar.gz
aws s3 cp --recursive /opt/ml/input/training s3://<bucket>/<prefix>/export
```
  
# Step2: upload notebook file, and run it one cell by one
* create one directory named transsion under ~
* upload the attachedjupyter notebooks file (https://amazon.awsapps.com/workdocs/index.html#/document/fc1a22a78ab3cbd4ae86e5641114ed8c4b30c351d2b1138e995f28c9cb43cf3a)to transsion directory
* modify export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" in wenet/run.sh according the using gpu instance type.
* add pip -r install requirements.txt
* open it, run one cell by one, pay attention on modifying data.list manually if needed.

Notebook environment:
Image:PyTorch 1.11/1.12 Python 3.8 CPU optimized
Kernel: Python3

Just for reference, on one p4d.24xlarge, the whole training time is 6h34min per epoch

# Step3: average_model & export model on  System terminal of SageMaker Studio
```
#download the model.tar.gz from S3 bucket of step2
aws s3 cp s3://<bucket>/<prefix>/model.tar.gz exp/sp_spec_aug/model.tar.gz
cd exp/sp_spec_aug
tar zxvf model.tar.gz
rm -fr model.tar.gz

#Change run.sh to execute step5
average_num=1 # this number should be the total count of *.pt files in your model.tar.gz except init.pt (http://init.pt/) file.
step =5
stop_step=5
./run.sh

#Change run.sh to execute step6
step =6
stop_step=6
./run.sh
```
