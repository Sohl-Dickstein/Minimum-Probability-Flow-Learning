function [state, cprobs] = sw_allall_bm(state, weights, biases, iters)
%SW_ALLALL_BM Swendsen-Wang algorithm for fully-connect Boltzmann machine
%
%     [state, cprobs] = sw_allall_bm(state, weights, biases, iters)
%
% Takes "iters" steps of the S-W Markov chain from state. The Boltzmann machine
% is a probability distribution over a vector of {0,1} variables. The
% distribution is a reparameterization of the classic {-1,+1} Ising model.
%
%     p(s) \propto \exp( -E(s;W,b) )
%
% Inputs:
%        state DxN Markov chain state: N independent D-dimensional vectors.
%                  Should be {0,1} variables. Use Ising functions for {+1,-1} vars.
%      weights DxD Every off-diagonal element weights(i,j) = W_{ij} contributes
%                      - W_{ij} s_i s_j
%                  to the energy. This means there are two terms for every (i,j)
%                  pair. To avoid "double counting" only put weight parameters
%                  into either the upper or the lower diagonal of the weights
%                  matrix, or put each weight parameter in both halves, but
%                  remember to divide them by two.
%       biases Dx1 or DxN
%                  The biases define the final part of the energy.
%                      -E = \sum_{i=1}^D \sum_{j=1}^D W_{ij} s_i s_j + \sum_i b_i s_i
%                  Each Markov chain can optionally have its own bias vector,
%                  which is useful in applications where the "bias" is an input
%                  from another layer of the model, different for each case.
%        iters 1x1 Number of Markov chain steps to take.
%
% Outputs:
%        state DxN Markov chain state after an update that leaves the
%                  Boltzmann machine distribution(s) invariant.
%       cprobs DxN Marginal probabilities that each site would have been set to
%                  the "1" state conditioned on the bond variables in the last update.

% Iain Murray, August 2007, November 2009

% Convert to Ising representation
ising_J = (weights+weights')/4;
ising_h = bsxfun(@plus, biases/2, sum(ising_J, 2));
ising_state = 2*state - 1;

% Run Ising code
[ising_state, cprobs] = sw_allall_ising(ising_state, ising_J, ising_h, iters);

% Convert back
state = (ising_state + 1) / 2;
