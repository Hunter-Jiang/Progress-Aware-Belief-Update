echo "### SFT ON EXISTING TRAJECTORIES ###"

# sft
for lr in 5e-6;
do
	echo "training on LR = $lr" 
	accelerate launch ../src/PABU_training.py \
		--base_model_name_or_path meta-llama/Llama-3.2-1B \
		--datafile_train HunterJiang97/PABU-Data \
		--train_col all \
		--save_path_format ../model/PABU-Agent-1b-2048-sft-e{}-s{}-c{}-lr{}/ \
		--lr $lr \
		--max_length 2048 \
		--num_epochs 1 \
		--batch_size 4
done
