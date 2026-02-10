#!/bin/bash

echo "Playing AgentGym on test split"
echo "Note: Tested on Ubuntu 20.04 LTS + python3.9"

# List of models to test
models=(
    "HunterJiang97/PABU-Agent-8B" # PABU checkpoint hosted on huggingface
)

# List of ports (one per environment)
ports=(36001 36102 36203 36304 36405 36405 36506 36607)

# Corresponding output directories for each environment
game_type=(
    "sciworld"
    "alfworld"
    "babyai"
    "textcraft"
    "maze"
    "wordle"
    "weather"
    "movie"
)

max_iter=(30 30 20 20 15 6 10 12)

task_file=(
    "../evaluation_split/sciworld_test.npy"
    "../evaluation_split/alfworld_test.npy"
    "../evaluation_split/babyai_test.npy"
    "../evaluation_split/textcraft_test.npy"
    "../evaluation_split/maze_test.npy"
    "../evaluation_split/wordle_test.npy"
    "../evaluation_split/weather_test.npy"
    "../evaluation_split/movie_test.npy"
)

# Outer loop: over models
for model in "${models[@]}"; do
    echo "Testing model: $model"
    
    # Inner loop: over environments (ports/output dirs)
    for i in "${!ports[@]}"; do
    #for ((i=${#ports[@]}-1; i>=0; i--)); do
        echo "  Running environment ${game_type[$i]} on port ${ports[$i]} saving to ${output_dirs[$i]}"
        
        accelerate launch ../src/PABU_evaluation.py \
            --task_file "${task_file[$i]}" \
            --base_model_name_or_path $model \
            --saved_path "" \
            --game_type "${game_type[$i]}" \
            --max_iter ${max_iter[$i]} \
            --play_save_path "../eval/${game_type[$i]}_${model//\//_}" \
            --max_length 2048 \
            --instances 2 \
            --batch_size 2 \
            --play_port "${ports[$i]}"
    done
done
