import sentencepiece as spm
import re
import os

## Path settings
base_path = os.getcwd()
project_name = "sample_run"
dataset_path = "/stek/corpora/"         # Define the dataset Full path here
log_path = os.path.join(base_path, "runs", project_name) # directory to store the logs
os.makedirs(log_path, exist_ok = True)
model_path = os.path.join(base_path, "trained_model", project_name) # directory to save the model
os.makedirs(model_path, exist_ok = True)
result_path = os.path.join(base_path, "results", project_name) # directory to save the results
os.makedirs(result_path, exist_ok = True)

model_filename =  "Conf_mod246_SLoss_Batch_Gating_0.5-tc1000" # "Conf_mod001_Batch_Gating_0.5"
test_model_path = os.path.join(base_path, "trained_model", "Gating_0.5_Libri1000", model_filename)    # Default model trained with Probabilistic gating of 0.5 

# Training and Inference time Variables
brv_prob = 0.5                     # For Training and Inference, Probability based on Bernoulli Random Variable for the output value to be 1. 
inf_drop_type = "greedy"    # can be "random_fixed" or "greedy". In case of greedy, set the gate_values in inference.py file (in generate_gate_values function)
blocks_to_drop = 6                 # For Inference, refers to "n" in the LDASR paper --- number of blocks to drop

bpe_flag = True
flag_use_single_out = True           # If True, use only one decoder, if False, operates like Early-Exit
flag_use_gating = True               # If True, Use gating mechanism, otherwise normal conformer model is used
flag_use_batch_gating = True         # If True, updates the gate valus for each batch

# GPU device setting
num_workers = 10
shuffle = True

# model parameter setting
batch_size = 48                     # batch size
max_len = 2000                      # maximum length in positional encoding 
d_model = 256                       # embedding dimension
n_encoder_layers = 2                # Attention blocks for each exit (for early-exit archietcture)
n_decoder_layers = 6                # Decoder Blocks
n_heads = 8                         # Attention heads
n_enc_replay = 6                    # How many times to repeat "n_encoder_layers"
dim_feed_forward = 2048             # (Up-sampling factor = 8) Conformer's Feed-Forward sub-block's dimension 
drop_prob = 0.1                     # Dropping probability
depthwise_kernel_size = 31          # Convolutional Kernel size in Conformer model
max_utterance_length = 401          # maximum utterance length to include utterance for training

## For Decoder
src_pad_idx = 0                     # Source padding token
trg_pad_idx = 30                    # Target Padding token
trg_sos_idx = 1                     # Start of Sentence Token
trg_eos_idx = 31                    # End of Sentence Token
enc_voc_size = 32                   # Encoder Vocabulary Size
dec_voc_size = 32                   # Decoder Vocabulary Size

sample_rate = 16000
n_fft = 512                         # 512-point fast fourier transform
win_length = 320                    # 20 ms chunks of audio
hop_length = 160                    # 10 ms overlap between chunks
n_mels = 80                         # number of mel=filter banks
n_mfcc = 80                         # number of mel-frequency cepstral coefficient 

# Sentence piece tokenizer
sp = spm.SentencePieceProcessor()
if bpe_flag == True:
    sp.load(os.path.join(base_path, 'libri.bpe-256.model'))     # BPE-256 model for LibriSpeech
    src_pad_idx = 0
    trg_pad_idx = 126
    trg_sos_idx = 1
    trg_eos_idx = 2
    enc_voc_size = sp.get_piece_size()
    dec_voc_size = sp.get_piece_size()
    lexicon = os.path.join(base_path, "librispeech-bpe-256.lex")      # BPE-256 lexicon for LibriSpeech
    tokens = os.path.join(base_path, "librispeech-bpe-256.tok")       # BPE-256 tokens for LibriSpeech

## optimizer parameter setting
init_lr = 1e-5
factor = 0.9
adam_eps = 1e-9
patience = 10

epoch = 500
clip = 1.0
weight_decay = 5e-4
inf = float('inf')
