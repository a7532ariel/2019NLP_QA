# python run_race.py --data_dir=RACE --bert_model=bert-large-uncased --output_dir=large_models --max_seq_length=321 --do_train --do_eval --do_lower_case --train_batch_size=8 --eval_batch_size=1 --learning_rate=1e-5 --num_train_epochs=2 --gradient_accumulation_steps=8 --fp16 --loss_scale=128 && /root/shutdown.sh

python test_race.py --data_dir=data/ --bert_model=bert-base-chinese --load_model=base_models/merge_pytorch_model_4.bin --output_name=result7.csv --max_seq_length=512  --do_eval --train_batch_size=6 --eval_batch_size=8 --learning_rate=1e-4 --num_train_epochs=15 --gradient_accumulation_steps=8 --loss_scale=128 

