#!/bin/bash

lr=0.01
keep_prob=1
data=cifar10
root=~/data
model=vgg
model_out=./checkpoint/${data}_${model}_sap
echo "model_out: " ${model_out}
CUDA_VISIBLE_DEVICES=2 python ./main_sap.py \
                        --lr ${lr} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
                        --keep_prob ${keep_prob} \
                        #> >(tee ${model_out}.txt) 2> >(tee error.txt)
