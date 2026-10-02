#!/usr/bin/env bash
# Train STATE (Arc's released Replogle-HVG configuration) on one source line, our split, GO-SVD perturbation features.
# usage: st_train.sh SRC SEED STEPS [NAME_SUFFIX]
SRC=$1; SEED=$2; STEPS=$3; SUF=${4:-}
cd /root/st
export WANDB_MODE=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
state tx train \
  data.kwargs.toml_config_path=/root/st/toml/${SRC}_s${SEED}.toml \
  data.kwargs.embed_key=X_hvg data.kwargs.output_space=gene data.kwargs.pert_col=gene \
  data.kwargs.cell_type_key=cell_line data.kwargs.batch_col=gem_group data.kwargs.control_pert=non-targeting \
  data.kwargs.perturbation_features_file=/root/st/feat.pt data.kwargs.num_workers=2 \
  model=state model.kwargs.cell_set_len=64 model.kwargs.hidden_dim=328 \
  model.kwargs.n_encoder_layers=1 model.kwargs.n_decoder_layers=1 model.kwargs.batch_encoder=true \
  model.kwargs.transformer_backbone_kwargs.max_position_embeddings=64 \
  model.kwargs.transformer_backbone_kwargs.hidden_size=328 \
  model.kwargs.transformer_backbone_kwargs.intermediate_size=3072 \
  model.kwargs.transformer_backbone_kwargs.num_hidden_layers=8 \
  model.kwargs.transformer_backbone_kwargs.num_attention_heads=4 \
  model.kwargs.transformer_backbone_kwargs.num_key_value_heads=4 \
  model.kwargs.transformer_backbone_kwargs.head_dim=82 \
  training.batch_size=64 training.lr=0.001 training.weight_decay=0.0005 training.max_steps=${STEPS} \
  training.val_freq=1000 training.ckpt_every_n_steps=2000 training.train_seed=42 training.wandb_track=false \
  use_wandb=false output_dir=/root/st/runs name=${SRC}_s${SEED}${SUF}
