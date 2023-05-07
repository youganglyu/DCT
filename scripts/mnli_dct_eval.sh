#export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME="/usr/local/cuda-11.0"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
python3 -u mnli_dct.py \
--data_dir PATH_TO_DATA_DIR/MNLI \
--model_name_or_path $1 \
--task_name mnli \
--output_dir PATH_TO_OUTPUT_DIR/ \
--do_eval \
--dct_per_gpu_eval_batch_size 32 \
--dct_per_gpu_train_batch_size 32 \
--dct_gradient_accumulation_steps 1 \
--dct_max_seq_length 256 \
--dct_max_steps 250 \
--dct_num_train_epochs 5 \
