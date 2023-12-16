#!/bin/bash
# run with 4 random seeds for each annotator with save hyperparameters

cd ..
source venv/bin/activate
budget=(0.25 0.5 0.75 1)
GPUs=() # set gpus
seed=0 

annotators_1=('0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23')


for ann_index in  "${!annotators_1[@]}" ; do
    for b_index in "${!budget[@]}" ; do
        tasks_1=${annotators_1[$ann_index]}
        b=${budget[$b_index]}
        # Get current gpu
        gpu_index=$(($b_index % ${#GPUs[@]}))
        gpu=${GPUs[$gpu_index]}
        
        echo "Running model for tasks: $tasks_1, gpu: $gpu"
        
        SESSION_NAME="gpu_${gpu}_budget_${b}_all_tasks"
        
        screen -dmS "$SESSION_NAME" bash -c "CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                        --dataset "mfrc" \
                                                        --label "Moral" \
                                                        --mtl_tasks "$tasks_1" \
                                                        --budget $b \
                                                        --run_sweep \
                                                        --baseline \
                                                        --train_batch_size 64 \
                                                        --seed $seed ;
                                            "
    done
    
done