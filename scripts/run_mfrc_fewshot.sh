#!/bin/bash
# run with 4 random seeds for each annotator with save hyperparameters

cd ..
source venv/bin/activate

GPUs=() 
dataset='mfrc'
label='Moral'
seed=0
sampling_method="random"
annotators_1=("0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23")

k_shots=("128,32,64,16")
n_mtls=(6 12 18) # 12  6 18


# Loop over the first half of tasks and run in parallel
for k_index in "${!k_shots[@]}" ; do

    k_shot=${k_shots[$k_index]}

    for ann_index in  "${!annotators_1[@]}" ; do

        few_shot_task_1=${annotators_1[$ann_index]}
        # few_shot_task_2=${annotators_2[$ann_index]}
        # few_shot_task_3=${annotators_3[$ann_index]}
        # few_shot_task_4=${annotators_4[$ann_index]}

        for n_index in  "${!n_mtls[@]}" ; do

            n_mtl=${n_mtls[$n_index]}

            # Get current gpu
            gpu_index=$(($n_index % ${#GPUs[@]}))
            gpu=${GPUs[$gpu_index]}

            echo "Running model for : $few_shot_task_1, on $n_mtl mtl tasks, with $k_shot shot on gpu: $gpu"

            SESSION_NAME="mfrc_gpu_${gpu}_sampling_${sampling_method}_seed_${seed}_n_mtl_${n_mtl}"

            screen -dmS "$SESSION_NAME" bash -c "CUDA_VISIBLE_DEVICES=$gpu python fewshot_main.py \
                                                                            --dataset "$dataset" \
                                                                            --label "$label" \
                                                                            --few_shot_task "$few_shot_task_1" \
                                                                            --n_mtl_tasks $n_mtl \
                                                                            --seed $seed \
                                                                            --balance_ratio 0 \
                                                                            --epochs 30 \
                                                                            --k_shot $k_shot \
                                                                            --few_shot_sample_strategy "$sampling_method" ;
                                                "
        done

    done

done