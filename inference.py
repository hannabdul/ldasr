import os
import time
import torch
import random
import torchaudio

from conf import *
from models.model.conformer_model import Conformer_with_LayerDrop

from transforms import *
from util.beam_infer import ctc_predict
from util.data_loader import collate_infer_fn
from util.tokenizer import apply_lex, load_dict

torch.set_printoptions(precision=5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1234)
random.seed(1234)

model = Conformer_with_LayerDrop(src_pad_idx = src_pad_idx,
                        n_enc_replay = n_enc_replay,
                        d_model = d_model,
                        enc_voc_size = enc_voc_size,
                        dec_voc_size = dec_voc_size,
                        max_len = max_len,
                        dim_feed_forward = dim_feed_forward,
                        n_head = n_heads,
                        n_encoder_layers = n_encoder_layers,
                        features_length = n_mels,
                        drop_prob = drop_prob,
                        depthwise_kernel_size = depthwise_kernel_size,
                        device = device,
                        flag_use_single_out = flag_use_single_out,
                        flag_use_gating = flag_use_gating).to(device)

def generate_gate_values(inf_drop_type, n = blocks_to_drop, brv_prob = brv_prob):
    if inf_drop_type == "probabilistic":
        # brv_prob --- Probability based on BRV for the gate_values to be equal to 1
        gate_values = torch.bernoulli(torch.randint(1, (12,)), p = brv_prob)
        scaling_factor = 1 / brv_prob
        # print("Using Probablistic dropping with BRV dropping prob {}".format(brv_prob))
        # print("The scaling factor is {}".format(scaling_factor))
    elif inf_drop_type == "random_fixed":
        # n -- Number of blocks to select (2, 4, 6, 8, 10, 12)
        gate_values = torch.zeros(1, 12)
        temp = torch.Tensor(random.sample(range(0, 12), n)).to('cuda', torch.long).reshape(1, -1)
        gate_values[torch.arange(gate_values.size(0)), temp.t()] = 1
        scaling_factor = round(12 / (12 - n), 3)
        # print("Using Random-Fixed dropping and dropping {} conformer blocks".format(12 - n))
        # print("The scaling factor is {}".format(scaling_factor))
    elif inf_drop_type == "greedy": # For Greedy dropping, set the gate_values here
        gate_values = torch.tensor([1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1]) # Define here
        n = torch.count_nonzero(gate_values).item()
        scaling_factor = 12 / (12 - n)
        # print("Using Greedy dropping and dropping {} conformer blocks".format(12 - n))
        # print("The scaling factor is {}".format(scaling_factor))
    gate_values = gate_values.reshape(-1)    # Shape = (12,)
    return gate_values, scaling_factor

def evaluate(model, inf_drop_type, blocks_to_drop, brv_prob):
    
    file_dict = os.path.join(base_path, "librispeech.lex")
    words = load_dict(file_dict)
     
    model.eval()
    
    set_ = "test-clean"
    if set_ == "test-clean":
    # for set_ in "test-clean","test-other","dev-clean", "dev-other":
        print(set_, ", Using Device: ", device)
        print("Batch Gating is : ", flag_use_batch_gating)
        
        test_dataset = torchaudio.datasets.LIBRISPEECH(dataset_path, 
                                                       url = set_, 
                                                       download = False)
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                  pin_memory = False, 
                                                  batch_size = 1, 
                                                  num_workers = 0, 
                                                  shuffle = False, 
                                                  collate_fn = collate_infer_fn)
        total_time = 0
        gate_values = 0
        scaling_factor = 0

        for it, batch in enumerate(test_loader):
            t_start = time.time()
            
            if flag_use_batch_gating:
                gate_values, scaling_factor = generate_gate_values(inf_drop_type, blocks_to_drop, brv_prob)
                model.set_gate_values(gate_values)
                
                if len(gate_values) != 12:
                    print("ERROR: Length of Gate Values should be 12. The current length is {}".format(len(gate_values)))
                    break
            
            trg_expect = batch[1][:,1:].to(device) #shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]   
            trg = batch[1][:,:-1].to(device) #cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28] 
            for trg_expect_ in trg_expect:
                if bpe_flag == True:
                    print(set_,"EXPECTED:",sp.decode(trg_expect_.squeeze(0).tolist()).lower())
                else:                    
                    print(set_,"EXPECTED:",re.sub(r"[#^$]+","",text_transform.int_to_text(trg_expect_.squeeze(0))))
            valid_lengths=batch[2]
            
            model_inp = batch[0].to(device)
            encoder = model(model_inp, valid_lengths)
            
            ## Scale Here
            encoder = scaling_factor * encoder
            
            if not flag_use_single_out:    # Use this if the model is trained with early-exits and you desire the output of only last exit
                i = 0
                for enc in encoder:
                    i = i + 1
                    best_combined = ctc_predict(enc, i - 1)
                    for best_ in best_combined:
                        if bpe_flag==True:
                            print(set_," BEAM_OUT_",i,":",  apply_lex(sp.decode(best_).lower(),words))
                        else:
                            print(set_," BEAM_OUT_",i,":",  apply_lex(re.sub(r"[#^$]+","",best_.lower()),words))
                t_end = time.time()
                # print("Total Time taken per batch: ", t_end - t_start)
                total_time = total_time + (t_end - t_start)
                # print("Total Time Taken: ", total_time)
            else:
                i = 1
                best_combined = ctc_predict(encoder[0], i - 1)

                for best_ in best_combined:
                    if bpe_flag == True:
                        print(set_," BEAM_OUT_",i,":",  apply_lex(sp.decode(best_).lower(),words))
                    else:
                        print(set_," BEAM_OUT_",i,":",  apply_lex(re.sub(r"[#^$]+","",best_.lower()),words))
                t_end = time.time()
                # print("Total Time taken per batch: ", t_end - t_start)
                total_time = total_time + (t_end - t_start)
            
        print("Total Time Taken: ", total_time, ' seconds')

###############################################################################
###############################################################################
if __name__ == '__main__':
    ## Model Trained on Libri-1000 with Batch_Gating of 0.8
    model.load_state_dict(torch.load(test_model_path, map_location = device))
    evaluate(model, inf_drop_type, blocks_to_drop, brv_prob)
    
    
