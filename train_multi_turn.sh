MODELPATH="./qwen"
DATAPATH="./m_data.json"
MODEL_SIZE="7B"
RUNNAME="longzhu"
OUTPUTPATH="./output1"
TOTALBSZ=1
BSZPERDEV=1
DEVICES="0"
NUMGPUS=$(echo $DEVICES | awk -F',' '{print NF}')
GRADACC=$(($TOTALBSZ/$NUMGPUS/$BSZPERDEV))
EPOCHNUM=8
echo "Training mistral model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BSZPERDEV batch size per GPU, $GRADACC gradient accumulation steps"

deepspeed --include localhost:$DEVICES --master_port 29502 ./train_multi_turn.py \
    --model_name_or_path ${MODELPATH} \
    --data_path ${DATAPATH} \
    --output_dir ${OUTPUTPATH}/${RUNNAME} \
    --num_train_epochs ${EPOCHNUM} \
    --per_device_train_batch_size ${BSZPERDEV} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADACC} \
    --eval_steps 50 \
    --save_strategy "no" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --do_eval False \
    --evaluation_strategy "no" \
    --model_max_length 256 \
    --conv_template "qwen" \
    --mask_user True \
    --run_name ${RUNNAME} \
    --bf16 True \
    --deepspeed ./deepspeed_config/deepspeed_config_zero2_no_offload.json
