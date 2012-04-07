%SW_ALLALL_FLIP_CONDITIONALS Swendsen-Wang helper: find+flip connected components
%
%     [flips, cprobs] = sw_allall_flip_conditionals(bonds, biases)
%
% Inputs:
%       bonds DxD binary bond variables in Random Cluster model used by S-W
%      biases Dx1 difference in log-probability contributed by this site for
%                 sticking rather than flipping.
%
% Outputs:
%       flips Dx1 {+1,-1} variables to multiply current state by to get new state.
%                 That is +1 is "don't flip" and -1 is "flip".
%      cprobs Dx1 marginal probabilities of flip=+1 conditioned on bonds.
%
% This routine is implemented as a mex file. See the comments in the C source
% for more information.

% Iain Murray, August 2007, November 2009
