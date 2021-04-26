## Tree_generation

# Abstract
Replicating urban patterns in urban designs is inherently complex and generally regarded as a qualitative task in many urban planning practices. Data-driven/ approaches are lacking as geometric modelling and sequence-based deep learning approaches cannot encapsulate the hierarchical nature of 3D building data. To overcome these limitations, we leverage on a proprietary tree structured neural network (“AETree”) developed by the AI for Civil Engineering Lab (AI4CE) which encodes and  reconstructs 3D building data. We suggest that these encoded features may be robust, expressive and objective representations of urban patterns in the real-world. Our capstone project aims to test this hypothesis by implementing AETree across New York City neighbourhoods. We will also experiment to overcome technical challenges such as increasing the decoder’s output length and using latent space interpolation to achieve realistic blended urban patterns.
# Baseline AETree model and its limitations
The AETree model discovers hierarchical structures in the areal spatial data. It applies a proprietary tree structured neural network to encode the original 3D building data and generate reconstructed 3D building data.

By learning the hierarchical structures within the data, the neural network understands the relationships between different scales of urban patterns (e.g. the relationship between a building and a cluster of buildings). This process strongly resembles how urban districts are created over time: the building morphology is determined relative to neighboring blocks, allowing a cluster of blocks to adopt similar forms. The encoder learns a latent representation of the original 3D building data, which can then be decoded to reconstruct the original data. Our team hypothesizes that this latent representation can be used as a standard, objective metric for evaluating urban patterns.

However, significant technical limitations remain that prevent the AETree from being used to derive a standard, objective metric for evaluating urban patterns:

- No evaluation was performed to confirm that the latent representation provides a meaningful, robust evaluation of a district’s urban patterns.
- The hierarchical clustering model is limited by the number of features it accepts as input, which is currently the building’s coordinates, length, width, height, and orientation angle. 
- The decoder module can only generate the same number of building blocks that the encoder was trained on (i.e. if 32 blocks were accepted as input, the decoder can only output 32 blocks).
- No evaluation was performed to validate whether latent space interpolation of two very different districts (e.g. Midtown Manhattan and central Paris) can result in a clearly blended (composite) urban pattern. 
