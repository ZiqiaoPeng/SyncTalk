dataset=May
workspace=model/trial_may
asr_model=ave

CUDA_VISIBLE_DEVICES=0 python main.py data/$dataset --workspace $workspace -O --iters 60000 --asr_model $asr_model --preload 1
CUDA_VISIBLE_DEVICES=0 python main.py data/$dataset --workspace $workspace -O --iters 100000 --finetune_lips --patch_size 64 --asr_model $asr_model --preload 1 
CUDA_VISIBLE_DEVICES=0 python main.py data/$dataset --workspace $workspace -O --test --asr_model $asr_model --portrait
