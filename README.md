# HARD-Lite: A Lightweight Hardware Anomaly Realtime Detection Framework Targeting Ransomware

#### This experiment demonstrates how performance counter data can be used to detect ransomware attacks using the LSTM anomaly detection model.

#### Prerequisite 
- Check your processor specification and use the appropriate version of [K-LEB tool](https://github.com/chutitepw/K-LEB).

- Install essential development tools and the kernel headers 
```
 apt-get install build-essential linux-headers-$(uname -r)
```

## Setup
- test.sh script can be used to automate all the steps below
  
### Apply the module (command line):

- Go to K-LEB folder and run the following commands:
```
make clean; make
sudo insmod kleb.ko
```

### Apply the module (with the script):
-  Run: 
```
sudo bash initialize.sh
```
- Select option: 2) Setup
    - The script will automatically insert K-LEB kernel module to the kernel.
      
### Training data collection

- To start monitoring using the kernel module, run the following bash command:
```
sudo ./ioctl_start -e <Event1>,<Event2>,<Event3>,<Event4>,<Event5>,<Event6> -t <timer interval (in ms)> -o <Log path> <program path>
```

- To setup K-LEB to collect overall system data during normal system operations, run the following bash command:
```
sudo ./ioctl_start -e 00c2,00c0,0729,0129,0229,ff9a -t 10 -o ../data/train/benign-train.csv -a
```
- For this experiment, K-LEB is set up to collect performance counter data from the processor in real-time and store it in .csv format. The sample data is collected in 10ms intervals to represent time series data. The hardware events used are listed below:

| Hardware Event* | Description | Event Number | Umask Value
| ------------- | ------------- | ------------- | ------------- |
| BR_RET  | Number of branch instructions retired.  | 0xc2 | 0x00
| INST_RET  | Number of instructions retired.  | 0xc0 | 0x00
| DCACHE_ACCESS | All Data Cache Accesses. | 0x29 | 0x07
| LOAD | Dispatch of a single op that performs a memory load. | 0x29 | 0x01
| STORE | Dispatch of a single op that performs a memory store. | 0x29 | 0x02
| MISS_LLC | L3 Misses by Request Type. | 0x9a | 0xff

*Different hardware events can be used but the performance may vary.

**This hardware event set is for AMD processors, please refer to the developer manual for Intel and ARM processors.

- After having sufficient data for training, stop collecting the data by pressing Ctrl+C or running the command:
```
sudo kill -INT $(pgrep ioctl_start);
```

### Train the classifier

- In the classifiers directory, run the following Python script to train the classifier:
```
python lstm-train.py
```
- The script will use the file in the directory /data/train/  to train the model.

### Test the classifier  
- In the classifiers directory, run the following Python script to use the existing test data sample to test the classifier:
```
python lstm-test.py
```
- The script will use the file in the directory /data/test/ to test the model.

| Files | Description |
| ------------- | ------------- |
| benign-*.csv  | File represents benign cases of the system. 
| ransom-*.csv  | Files represent different ransomware family's behaviors.


### Deploy the classifier  
- In K-LEB folder, run the following command to collect the performance counter data:
```
sudo ./ioctl_start -e 00c2,00c0,0729,0129,0229,ff9a -t 10 -o ../data/real-time/Output.csv -a &
```
- Then in the classifiers directory, run the following Python script to process the data sample and classify the data in real-time:
```
python lstm-deploy.py
```
- The script will use the Output.csv file in the directory /data/real-time/ to perform the classification.

## Unload the Module
### Unload with Command Line

- The K-LEB kernel module can be unloaded by running the following commands in the K-LEB directory:
```
make clean
sudo rm /dev/kleb
sudo rmmod kleb
```

### Unload with Script

- Run the script with:
```
sudo bash initialize.sh
```

- Select option: 2) Clean
    - The script will automatically unload K-LEB kernel module from the kernel.
      
# Citing

For more technical details please refer to the following paper:
```
@ARTICLE{10208245,
  author={Woralert, Chutitep and Liu, Chen and Blasingame, Zander},
  journal={IEEE Transactions on Circuits and Systems I: Regular Papers}, 
  title={HARD-Lite: A Lightweight Hardware Anomaly Realtime Detection Framework Targeting Ransomware}, 
  year={2023},
  volume={70},
  number={12},
  pages={5036-5047},
  keywords={Ransomware;Monitoring;Behavioral sciences;Hardware;Servers;Registers;Operating systems;Performance evaluation;Semisupervised learning;Anomaly detection;Performance monitoring counters;semi-supervised learning;ransomware;anomaly detection;malware analysis},
  doi={10.1109/TCSI.2023.3299532}}
```
