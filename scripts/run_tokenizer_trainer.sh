# deepspeed --bind_cores_to_rank train_bert_ds.py --checkpoint_dir experiment_deepspeed $@
ps aux | grep pretrain_llama_ds | awk '{print $2}' | xargs -i kill -9 {}
deepspeed \
        --hostfile /workspace/projects/llm/config/hostfile \
        /workspace/projects/llm/src/llama/trainer_modeling_llama.py \
        --checkpoint_dir /workspace/projects/llm/data/check_points/deepspeed_llama \
        --train_datas_path_pattern "/workspace/projects/Open-Llama/data/pretrain_data/part-*.jsonl.zst" \
        --check_point_steps 100