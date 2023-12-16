#!/bin/bash
# run with 4 random seeds for each annotator with save hyperparameters

cd ..
source venv/bin/activate

GPUs=(1) # set gpus
seed=0 # 1,2
annotators_1=('4,6,13,15,17,18'
'5,7,8,16,18,20'
 '0,2,6,10,16,23'
 '2,4,10,14,15,20', '12,15,16,18,20,21')


annotators_2=(
 '8,9,14,15,16,21'
 '4,10,17,18,22,23'
 '2,11,17,18,20,23' ,'7,10,12,17,18,20'
 '3,10,11,12,15,23')

 annotators_3=(
 '1,9,10,20,21,23'
 '10,12,15,16,18,21', '1,2,5,15,19,21'
 '2,6,13,16,17,22'
 '1,2,6,7,12,18')

 annotators_4=(
 '0,5,13,18,21,23','2,4,6,7,9,19'
 '1,2,9,14,16,21'
 '3,4,6,8,11,20'
 '0,6,8,12,14,19')

#  annotators_5=(
#  )


# Loop over the first half of tasks and run in parallel
for ann_index in  "${!annotators_1[@]}" ; do
    
    tasks_1=${annotators_1[$ann_index]}
    tasks_2=${annotators_2[$ann_index]}
    tasks_3=${annotators_3[$ann_index]}
    tasks_4=${annotators_4[$ann_index]}
    tasks_5=${annotators_5[$ann_index]}

    # Get current gpu
    gpu_index=$(($ann_index % ${#GPUs[@]}))
    gpu=${GPUs[$gpu_index]}
    
    echo "Running model for tasks: $tasks_1, gpu: $gpu"
    
    SESSION_NAME="${gpu}_${tasks_1}"
    
    screen -dmS "$SESSION_NAME" bash -c "CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                    --dataset "mfrc" \
                                                    --label "Moral" \
                                                    --train_batch_size 64 \
                                                    --budget 0.25 \
                                                    --mtl_tasks "$tasks_1" \
                                                    --run_sweep \
                                                    --seed $seed ;
                                        CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                    --dataset "mfrc" \
                                                    --label "Moral" \
                                                    --train_batch_size 64 \
                                                    --budget 0.25 \
                                                    --mtl_tasks "$tasks_2" \
                                                    --run_sweep \
                                                    --seed $seed ;
                                        CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                   --dataset "mfrc" \
                                                    --label "Moral" \
                                                    --train_batch_size 64 \
                                                    --budget 0.25 \
                                                    --mtl_tasks "$tasks_3" \
                                                    --run_sweep \
                                                    --seed $seed ;
                                        CUDA_VISIBLE_DEVICES=$gpu python mtl_main.py \
                                                    --dataset "mfrc" \
                                                    --label "Moral" \
                                                    --train_batch_size 64 \
                                                    --budget 0.25 \
                                                    --mtl_tasks "$tasks_4" \
                                                    --run_sweep \
                                                    --seed $seed ;
                                        "
done