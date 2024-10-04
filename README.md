# Layer Dropping in Automatic Speech Recognition
This repository contains the official implementation of the paper entitled **"LDASR: An Experimental Study on Layer Drop using Conformer-based Architecture"** accepted at EUSIPCO-2024, Lyon, France.

# Summary 
In this paper, an exhaustive analysis is performed on Conformer based model to drop different number of conformer blocks using probability based dropping to find the optimum trade-off between the output's Word Error Rate (WER) and required computation. This work applies the Layer Drop (which itself is inspired from Stochastic Depth) technique to the Conformer architecture consisting of a shallow convolutional feature extractor (2 1-d convolution layers with 256 filters each), followed by 12 conformer blocks, a linear layer and a softmax layer. The architecture of Conformer module is modified by adding a Gating mechanism that controls the input flow in conformer submodules. If the value of the gate = 0, the input skips the Conformer sub-modules and only pass through the Layer Normalization at the very end while the input goes through each sub-block for gate = 1.
![Architecture 2](https://github.com/user-attachments/assets/30077656-6cfd-4d97-8f9d-c470076d6765)

The Conformer model is trained for different dropping probabilities (0.2, 0.4, 0.5, 0.6, 0.8) which defines how many conformer blocks will be used for training for current batch. The gate values (either 0 or 1) are inversely related to the dropping probability for each conformer block.

'''

'''

### Note: The code was originally written by [SpeechTeK Lab](https://github.com/SpeechTechLab) at Fondazione Bruno Kessler, Trento, Italy. This repo contains cleaned and commented version of the code with addition of gating mechanism to the conformer architecture as well as many duplicate variables removed.
