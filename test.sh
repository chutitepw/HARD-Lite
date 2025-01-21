#!/bin/bash

## This script will run through all HARD-Lite test cycle

## 1. Install K-LEB kernel module
cd ./K-LEB

make clean; make
sudo insmod kleb.ko

## 2. Collect HPC data with K-LEB for 10 seconds
 
sudo ./ioctl_start -e 00c2,00c0,0729,0129,0229,ff9a -t 10 -o ../data/train/benign-train.csv -a &

sleep 300

sudo kill -INT $(pgrep ioctl);

## 3. Train the classifier model using LSTM

cd ../classifiers

python lstm-train.py

## 4. Test the classifier model

python lstm-test.py

## 5 Deploy the classifier model

cd ../K-LEB

sudo ./ioctl_start -e 00c2,00c0,0729,0129,0229,ff9a -t 10 -o ../data/real-time/Output.csv -a &

sleep 5

cd ../classifiers

python lstm-deploy.py

sudo kill -INT $(pgrep ioctl);