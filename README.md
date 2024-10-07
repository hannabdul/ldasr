# Layer Dropping in Automatic Speech Recognition
This repository contains the official implementation of the paper entitled **"LDASR: An Experimental Study on Layer Drop using Conformer-based Architecture"** accepted at EUSIPCO-2024, Lyon, France. The paper targets creating dynamic models for Automatic Speech Recognition.

# Summary 
In this paper, an exhaustive analysis is performed on Conformer based architecture to find how many conformer blocks could be dropped using probability based layer dropping without compromising much on the final output (trade-off between output's Word Error Rate (WER) and computational power). This work applies the Layer Drop (which itself is inspired from Stochastic Depth) technique to the Conformer architecture consisting of a shallow convolutional feature extractor (2 1-d convolution layers with 256 filters each), followed by 12 conformer blocks, a linear layer and a softmax layer. The architecture of Conformer module is modified by adding a Gating mechanism that controls the input flow in conformer submodules. If the value of the gate = 0, the input skips the Conformer sub-modules and only pass through the Layer Normalization at the very end while the input goes through each sub-block for gate = 1.
![Architecture 2](https://github.com/user-attachments/assets/30077656-6cfd-4d97-8f9d-c470076d6765)

The Conformer model is trained for different dropping probabilities p_d (0.2, 0.4, 0.5, 0.6, 0.8) which defines how many conformer blocks will be used for training for current batch. The gate values (either 0 or 1) are inversely related to the dropping probability for each conformer block. Our findings include 
- The dropping probability = 0.5 is the optimum trade-off between WER and computational complexity. Meaning, at inference time, the model trained with p_d = 0.5 results in the best dynamic model that gives the output with minimum WER when tested in various resource settings (low, normal, and high). 
- Retaining the last Normalization Layer in Conformer block during training, helps the model to converge quickly and the training curve is smoother as compared to when the Normalization Layer is also skipped. ![Ablation - Layer Norm](https://github.com/user-attachments/assets/dc2862a5-0a68-4790-8dee-e2e448015608)

Trained Model with p_d = 0.5 is available [here](https://drive.google.com/drive/folders/1-2awgUupRqTJnPxXmScfqSlWLK-6qN8d?usp=sharing ) 

# How to Use the Repo
The most important file is the **conf.py** that contains all the configuration for the training and inference phase.

### Note: The code was originally written by [SpeechTeK Lab](https://github.com/SpeechTechLab) at Fondazione Bruno Kessler, Trento, Italy. This repo contains cleaned and commented version of the code with addition of gating mechanism to the conformer architecture.
