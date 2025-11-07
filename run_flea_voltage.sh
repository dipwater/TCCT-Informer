# train

# Normal MS
python -u main_informer.py --model informer --data Normal --root_path "./data/FLEA/" --data_path "Normal_filtered1.csv" --des 'Exp1' --freq 's' --target "Motor Y Voltage" --features MS --seq_len 60 --label_len 30 --pred_len 20 --e_layers 2 --d_layers 1 --n_heads 8 --d_model 512 --dropout 0.05 --attn 'prob' --itr 1 --train_epochs 50 --batch_size 32

# Normal S
python -u main_informer.py --model informer --data Normal --root_path "./data/FLEA/" --data_path "Normal_filtered1.csv" --des 'Exp1' --freq 's' --target "Motor Y Voltage" --features S --seq_len 60 --label_len 30 --pred_len 20 --e_layers 2 --d_layers 1 --n_heads 8 --d_model 512 --dropout 0.05 --attn 'prob' --itr 1 --train_epochs 50 --batch_size 32

# Jam MS
python -u main_informer.py --model informer --data Jam --root_path "./data/FLEA/" --data_path "Jam_filtered.csv" --des 'Exp1' --freq 's' --target "Motor Y Voltage" --features MS --seq_len 60 --label_len 30 --pred_len 20 --e_layers 2 --d_layers 1 --n_heads 8 --d_model 512 --dropout 0.05 --attn 'prob' --itr 1 --train_epochs 50 --batch_size 32

# Jam S
python -u main_informer.py --model informer --data Jam --root_path "./data/FLEA/" --data_path "Jam_filtered.csv" --des 'Exp1' --freq 's' --target "Motor Y Voltage" --features S --seq_len 60 --label_len 30 --pred_len 20 --e_layers 2 --d_layers 1 --n_heads 8 --d_model 512 --dropout 0.05 --attn 'prob' --itr 1 --train_epochs 50 --batch_size 32

# Position MS
python -u main_informer.py --model informer --data Position --root_path "./data/FLEA/" --data_path "Position_filtered.csv" --des 'Exp1' --freq 's' --target "Motor Y Voltage" --features MS --seq_len 60 --label_len 30 --pred_len 20 --e_layers 2 --d_layers 1 --n_heads 8 --d_model 512 --dropout 0.05 --attn 'prob' --itr 1 --train_epochs 50 --batch_size 32

# Position S
python -u main_informer.py --model informer --data Position --root_path "./data/FLEA/" --data_path "Position_filtered.csv" --des 'Exp1' --freq 's' --target "Motor Y Voltage" --features S --seq_len 60 --label_len 30 --pred_len 20 --e_layers 2 --d_layers 1 --n_heads 8 --d_model 512 --dropout 0.05 --attn 'prob' --itr 1 --train_epochs 50 --batch_size 32

# Spall MS
python -u main_informer.py --model informer --data Spall --root_path "./data/FLEA/" --data_path "Spall_filtered.csv" --des 'Exp1' --freq 's' --target "Motor Y Voltage" --features MS --seq_len 60 --label_len 30 --pred_len 20 --e_layers 2 --d_layers 1 --n_heads 8 --d_model 512 --dropout 0.05 --attn 'prob' --itr 1 --train_epochs 50 --batch_size 32

# Spall S
python -u main_informer.py --model informer --data Spall --root_path "./data/FLEA/" --data_path "Spall_filtered.csv" --des 'Exp1' --freq 's' --target "Motor Y Voltage" --features S --seq_len 60 --label_len 30 --pred_len 20 --e_layers 2 --d_layers 1 --n_heads 8 --d_model 512 --dropout 0.05 --attn 'prob' --itr 1 --train_epochs 50 --batch_size 32






# pred
python -u main_informer.py --model informer --data Normal --root_path "./data/test/" --data_path "Normal_filtered.csv" --des 'Exp1' --do_predict --freq 's' --target "Motor Y Voltage" --features MS --seq_len 60 --label_len 30 --pred_len 20 --e_layers 2 --d_layers 1 --n_heads 8 --d_model 512 --dropout 0.05 --attn 'prob' --itr 1 --train_epochs 50 --batch_size 32

