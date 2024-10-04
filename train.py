import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, AdamW

from conf import *
from transforms import *
from dataloaders import *

from models.model.conformer_model import Conformer_with_LayerDrop
from models.model_functions import count_parameters, initialize_weights
from NoamOpt import NoamOpt
from util.epoch_timer import epoch_time
from util.data_loader import text_transform
from util.beam_infer import ctc_predict, greedy_decoder
from util.data_loader import collate_fn

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_path)

torch.manual_seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = Conformer_with_LayerDrop(src_pad_idx=src_pad_idx,
                        n_enc_replay=n_enc_replay,
                        d_model=d_model,
                        enc_voc_size=enc_voc_size,
                        dec_voc_size=dec_voc_size,
                        max_len=max_len,
                        dim_feed_forward=dim_feed_forward,
                        n_head=n_heads,
                        n_encoder_layers=n_encoder_layers,
                        features_length=n_mels,
                        drop_prob=drop_prob,
                        depthwise_kernel_size=depthwise_kernel_size,
                        device=device,
                        flag_use_single_out = flag_use_single_out,
                        flag_use_gating = flag_use_gating).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
warmup = 8000    # Untill warmup steps, the learning rate will increase linearly, followed by exponential decrease.
ctc_loss = nn.CTCLoss(blank = 0, zero_infinity = True)
optimizer = NoamOpt(d_model, warmup, AdamW(params = model.parameters(), lr = 0, betas = (0.9, 0.98), eps=adam_eps, weight_decay = weight_decay))
print("batch_size:",batch_size," num_heads:",n_heads," num_encoder_layers:", n_encoder_layers, " optimizer:","NOAM[warmup ",warmup, "]","vocab_size:",dec_voc_size,"SOS,EOS,PAD",trg_sos_idx,trg_eos_idx,trg_pad_idx,"data_loader_len:",len(train_loader),"DEVICE:",device) 

def train(model, iterator, ep_num, brv_prob, flag_use_single_out):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        if not batch:
            continue
        
        # # Update the Gate Probability per batch
        if flag_use_batch_gating:
            if i % 300 == 0:
                model.train_generate_gate_values(brv_prob)
                model.print_gate_values()
            else:
                model.train_generate_gate_values(brv_prob)
        
        src = batch[0].to(device) 
        trg = batch[1][:,:-1].to(device) # cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28] 
        trg_expect = batch[1][:,1:].to(device) # shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]   
        valid_lengths = batch[3]

        encoder = model(src, valid_lengths)
        
        ctc_target_len = batch[2]
        loss_layer = 0
        
        if i % 300 == 0:
            if bpe_flag == True:
                print("EXPECTED:",sp.decode(trg_expect[0].tolist()).lower())
            else:
                print("EXPECTED:",text_transform.int_to_text(trg_expect[0]))
        
        last_probs = encoder[encoder.size(0)-1].to(device)
        
        ctc_input_len = torch.full(size=(encoder.size(1),), fill_value = encoder.size(2), dtype=torch.long)
        # print(encoder.size(), ctc_input_len)

        if not flag_use_single_out:
            for enc in  encoder[0:encoder.size(0) - 1]:   # For Early-Exits
                #print(enc.size(),last_probs.size())
                loss_layer += ctc_loss(enc.permute(1,0,2),batch[1],ctc_input_len,ctc_target_len).to(device)
    
                if i % 300 == 0:
                    if bpe_flag==True:
                        print("CTC_OUT at [", i,"]:", sp.decode(ctc_predict(enc[0].unsqueeze(0))).lower())
                    else:
                        print("CTC_OUT at [", i,"]:", ctc_predict(enc[0].unsqueeze(0)))
            del encoder
            loss_layer += ctc_loss(last_probs.permute(1,0,2), batch[1], ctc_input_len, ctc_target_len).to(device)
        else:           ## For Using single decoder at the end
            loss_layer = ctc_loss(last_probs.permute(1,0,2), batch[1], ctc_input_len, ctc_target_len).to(device)
                    
        if i % 300 == 0:
            if bpe_flag==True:
                print("CTC_OUT at [", i,"]:", sp.decode(ctc_predict(last_probs[0].unsqueeze(0))).lower())
            else:
                print("CTC_OUT at [", i,"]:", ctc_predict(last_probs[0].unsqueeze(0)))
        
        loss = loss_layer

        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        writer.add_scalar('loss', loss.item())
        print('Epoch: ', ep_num, ', step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())
    
    return epoch_loss / len(iterator)

def run(model, total_epoch, best_loss):
    prev_loss = 9999999
    nepoch = 1
    best_model = os.path.join(model_path, 'Conf_mod{:03d}_Batch_Gating_{}'.format(nepoch, brv_prob))
    best_lr = os.path.join(model_path, 'Conf_lr{:03d}_Batch_Gating_{}'.format(nepoch, brv_prob))

    if os.path.exists(best_model):
        initialize_model=False
        print('loading model checkpoint: ', best_model)
        model.load_state_dict(torch.load(best_model, map_location=device))

    if os.path.exists(best_lr):
        print('loading learning rate checkpoint: ', best_lr)
        optimizer.load_state_dict(torch.load(best_lr))

    for step in range(nepoch, total_epoch):
        start_time = time.time()
        
        total_loss = train(model = model,                # Input Model
                            iterator = train_loader,     # Dataloader object for LibriSpeech dataset
                            ep_num = step,               # Current Epoch
                            brv_prob = brv_prob,         # Dropping probability
                            flag_use_single_out = flag_use_single_out) # Flag indicating whether to use single-exit (True) or early-exits(False)

        print("TOTAL_LOSS-", step,":=", total_loss)
        
        writer.add_scalar("Training Loss/Epoch", total_loss, step)
        
        if total_loss < prev_loss:
            prev_loss = total_loss
            best_model = os.path.join(model_path, 'Conf_mod{:03d}_Batch_Gating_{}'.format(step, brv_prob))
            
            print("saving:",best_model)
            torch.save(model.state_dict(), best_model)
            lrate = os.path.join(model_path, 'Conf_lr{:03d}_Batch_Gating_{}'.format(step, brv_prob))
            print("Saving:", lrate)
            torch.save(optimizer.state_dict(), lrate)
            print("Time per epoch : ", time.time() - start_time, " seconds")
        else:
            worst_model = os.path.join(model_path, 'Conf_mod{:03d}_Batch_Gating_{}'.format(step, brv_prob))
            print("WORST: not saving: ", worst_model)



if __name__ == "__main__":
    print("Using GPU : ", torch.cuda.get_device_name())
    print("Using GPU Number: ", torch.cuda.current_device())
    run(model = model, total_epoch = epoch, best_loss = inf)
    writer.flush()    
    print("closing the Writer...!!")