# Layer Dropping in Automatic Speech Recognition
This repository contains the official implementation of the paper entitled **"LDASR: An Experimental Study on Layer Drop using Conformer-based Architecture"** accepted at EUSIPCO-2024, Lyon, France.

# Summary 
This work applies the Layer Drop (which itself is inspired from Stochastic Depth) technique to the Conformer based architecture. The overall architecture consists of a shallow convolutional feature extractor (2 1-d convolution layers with 256 filters each), followed by 12 conformer blocks. We modified the Conformer module's architecture by adding a Gating unit that controls the input flow in conformer submodules.

'''

'''

> @inproceedings{hannan2024,
    author = {Hannan, Abdul and Brutti, Alessio and Falavigna, Daniele},
    title = {{LDASR}: An Experimental Study on Layer Drop using Conformer-based Architecture},
    booktitle = {Proc. of EUSIPCO},
    year = {2024}
}

### Note: The code was originally written by [SpeechTeK Lab](https://github.com/SpeechTechLab) at Fondazione Bruno Kessler, Trento, Italy. This repo contains cleaned and commented version of the code with many duplicate variables removed.
