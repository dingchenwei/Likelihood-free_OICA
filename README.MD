# Likelihood-Free OICA
Existing OICA algorithms usually make strong parametric assumptions on the distribution of independent components, which may be violated on real data, leading to sub-optimal or even wrong solutions.

Likelihood-Free Overcomplete ICA algorithm (LFOICA) estimates the mixing matrix directly by backpropagation without any explicit assumptions on the density function of independent components. It first transform random noise into components, then use a generation process that mimic the mixing procedure that mix components into mixtures. MMD between the observed mixtures and estimated mixtures is used as a teacher to guide the mixing matrix and components learning process.

## Required python packages
pytorch

numpy

## Usage
python LFOICA.py --num_components --num_mixtures --cuda(if applicable)

