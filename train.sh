CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batch-size 2 --num-gpus 8\
                --ckpt-path output/unet_train/b6bf34xg/checkpoints/epoch=6-step=740000.ckpt\
                --pretransform-ckpt-path ""\
                --dataset-config stable_audio_tools/configs/dataset_configs/encodec_dataset.json \
                --model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_1_0_dac.json \
                --name unet_train


# python train.py --batch-size 8 --num-gpus 8\
#                 --pretransform-ckpt-path output/unet_train/fs2ul7m9/checkpoints/epoch=5-step=700000.ckpt\
#                 --dataset-config stable_audio_tools/configs/dataset_configs/local_training_example.json \
#                 --model-config stable_audio_tools/configs/model_configs/txt2audio/stable_audio_1_0_dac.json \
#                 --name harmonai_train


# python3 unwrap_model.py --model-config /path/to/model/config \
#                         --ckpt-path /path/to/wrapped/ckpt \
#                         --name model_unwrap

# find $PWD -maxdepth 1 | head -n 5000 > /mmu-audio-ssd/frontend/audioSep/wanghualei/code/stable-audio-tools/kwai.txt
# nohup python train.py >>unet95.log &
# nohup python train.py >>dit.log &
# nohup python inference.py >infer.log &
# ps -ef | grep train.py | grep -v grep | awk '{print $2}' | xargs kill -9
