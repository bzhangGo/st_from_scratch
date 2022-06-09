#! /bin/bash

export CUDA_VISIBLE_DEVICES=0

data=path-to-preprocessed-ende-dataset/
code=path-to-zero-codebase/
moses=path-to-mosesdecoder/

# average best 10 checkpoints
python3 ${code}/scripts/checkpoint_averaging.py --path ../train/best --output avg --checkpoints 10 --gpu 0

python3 ${code}/run.py --mode test --parameters=hidden_size=256,embed_size=256,filter_size=4096,\
dropout=0.1,label_smooth=0.1,attention_dropout=0.1,relu_dropout=0.2,residual_dropout=0.2,\
max_text_len=256,max_frame_len=480000,batch_size=80,eval_batch_size=35,\
token_size=20000,batch_or_token='token',\
initializer="uniform_unit_scaling",initializer_gain=0.5,beam_size=8,decode_alpha=1.4,\
model_name="transformer",scope_name="transformer",buffer_size=5000,data_leak_ratio=0.1,\
input_queue_size=1000,output_queue_size=1000,\
deep_transformer_init=True,\
audio_num_mel_bins=40,audio_add_delta_deltas=True,pdp_r=512,\
sinusoid_posenc=True,max_poslen=20480,ctc_enable=True,ctc_alpha=0.3,audio_dither=0.0,\
enc_localize="pdp",dec_localize="none",encdec_localize="none",\
clip_grad_norm=0.0,\
num_heads=4,\
process_num=4,\
lrate=1.0,\
estop_patience=100,\
num_encoder_layer=12,\
num_decoder_layer=6,\
warmup_steps=4000,\
lrate_strategy="noam",\
epoches=5000,\
update_cycle=25,\
gpus=[0],\
disp_freq=1,\
eval_freq=1000,\
save_freq=2500,\
sample_freq=1000,\
checkpoints=10,\
best_checkpoints=10,\
max_training_steps=50000,\
beta1=0.9,\
beta2=0.98,\
random_seed=1234,\
src_vocab_file="$data/vocab.zero.en",\
tgt_vocab_file="$data/vocab.zero.de",\
src_train_path="$data/en-de/data/train/wav/",\
src_train_file="$data/en-de/data/train/txt/train.yaml",\
tgt_train_file="$data/train.bpe.de",\
src_dev_path="$data/en-de/data/dev/wav/",\
src_dev_file="$data/en-de/data/dev/txt/dev.yaml",\
tgt_dev_file="$data/dev.bpe.de",\
src_test_path="$data/en-de/data/tst-COMMON/wav/",\
src_test_file="$data/en-de/data/tst-COMMON/txt/tst-COMMON.yaml",\
tgt_test_file="$data/test.bpe.de",\
output_dir="avg",\
test_output="trans.txt",\

# post processing
sed -r 's/ \@(\S*?)\@ /\1/g' < trans.txt |
sed -r 's/\@\@ //g' |
sed "s/&lt;s&gt;//" |
${moses}/scripts/recaser/detruecase.perl > trans.tok.txt

# evaluation
${moses}/scripts/generic/multi-bleu.perl $data/test.reftok.de < trans.tok.txt > test.bleu

# note to perform sacrebleu, you need `${moses}/scripts/tokenizer/detokenizer.perl -l de` to get the detokenized outputs