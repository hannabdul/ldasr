# Layer Dropping in Automatic Speech Recognition
This repository contains the official implementation of the paper entitled **"LDASR: An Experimental Study on Layer Drop using Conformer-based Architecture"** accepted at EUSIPCO-2024, Lyon, France.

# Summary 
This work applies the Layer Drop (which itself is inspired from Stochastic Depth) technique to the Conformer based architecture. The overall architecture consists of a shallow convolutional feature extractor (2 1-d convolution layers with 256 filters each), followed by 12 conformer blocks, a linear layer and a softmax layer. The architecture of Conformer module is modified by adding a Gating mechanism that controls the input flow in conformer submodules. If the value of the gate = 0, the input skips the Conformer sub-modules and only pass through the Layer Normalization at the very end while the input goes through each sub-block for gate = 1.


'''

'''

### Note: The code was originally written by [SpeechTeK Lab](https://github.com/SpeechTechLab) at Fondazione Bruno Kessler, Trento, Italy. This repo contains cleaned and commented version of the code with addition of gating mechanism to the conformer architecture as well as many duplicate variables removed.
