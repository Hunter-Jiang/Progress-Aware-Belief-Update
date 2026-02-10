#!/bin/bash

pip install -r requirements.txt
cd ./Toolusage

pip install -r requirements.txt

cd toolusage
pip install -e .
cd ..
cd ..
pip install --upgrade openai
pip install -e .

current_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJECT_PATH="$current_dir/Toolusage"
export MOVIE_KEY="eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJlZTBmYTQ1YjIxZmQ1Y2ZhMzI1Nzk2YmVmNTJiNDM0ZSIsIm5iZiI6MTc2NTkxODkyOS4xMDksInN1YiI6IjY5NDFjOGQxMmFiNDc5NmQ4MjNmODMzZCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.yH8EARPE4yXUV_GWS0OOL1nVbjSK9uuI0cyoYb9oyU8"
export TODO_KEY="39f9ccd0a2d587ae16ee44d747b4e3b784e156ae"
export SHEET_EMAIL="hjiang24@ncu.edu"