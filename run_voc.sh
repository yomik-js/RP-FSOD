#!/usr/bin/env bash
EXP_NAME=$1
SAVE_DIR=checkpoints/${EXP_NAME}
IMAGENET_PRETRAIN=weight/R-101.pkl                             # <-- change it to you path
IMAGENET_PRETRAIN_TORCH=weight/resnet101-5d3b4d8f.pth  # <-- change it to you path
SPLIT_ID=$2

#------------------------------ create config ------------------------------------ #
for shot in 3 5 10 20
do
    for seed in 1 2 3 4 5 
    do
        python3 tools/create_config.py --dataset voc --config_root configs/voc \
                 --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
    done
done


#------------------------------- Base Pre-train ---------------------------------- #
python3 main.py --num-gpus 1 --config-file configs/voc/defrcn_det_r101_base1.yaml     \
    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                                   \
           OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base1


# ------------------------------ Model Preparation -------------------------------- #
python3 tools/model_surgery.py --dataset voc --method remove                                    \
    --src-path ${SAVE_DIR}/defrcn_det_r101_base1/model_final.pth                      \
    --save-dir ${SAVE_DIR}/defrcn_det_r101_base1
BASE_WEIGHT=${SAVE_DIR}/defrcn_det_r101_base1/model_reset_remove.pth


# ------------------------------ Novel Fine-tuning -------------------------------- #
for shot in 3 5 10 20 
do
    for seed in 1 2 3 4 5 6 7 8 9 10
    do
        for repeat_id in 1 2 3 4 5
        do
            CONFIG_PATH=configs/voc/defrcn_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
            OUTPUT_DIR=${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/tfa-like${seed}/${shot}shot_seed${seed}_repeat${repeat_id}
            python3 main.py --num-gpus 1 --config-file ${CONFIG_PATH}                          \
                    --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                   \
                            TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
        done
    done
done

python3 tools/extract_results.py --res-dir ${SAVE_DIR}/defrcn_gfsod_r101_novel1/tfa-like1  # surmarize all results
