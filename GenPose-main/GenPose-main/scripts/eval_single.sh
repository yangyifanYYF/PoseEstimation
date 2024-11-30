CUDA_VISIBLE_DEVICES=0 python /workspace/code/GenPose-main/GenPose-main/runners/evaluation_single.py \
--score_model_dir /workspace/data/results/ckpts/ScoreNet/ckpt_epoch80.pth \
--energy_model_dir /workspace/data/results/ckpts/EnergyNet/ckpt_epoch49.pth \
--data_path /workspace/data \
--sampler_mode ode \
--max_eval_num 1000000 \
--percentage_data_for_test 1.0 \
--batch_size 256 \
--seed 0 \
--test_source real_test \
--result_dir /workspace/data/results \
--eval_repeat_num 50 \
--pooling_mode average \
--ranker energy_ranker \
--T0 0.55 \
# --save_video \
