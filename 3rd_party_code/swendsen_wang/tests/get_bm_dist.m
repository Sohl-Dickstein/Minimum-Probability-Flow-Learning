function [states, probs] = get_bm_dist(weights, biases)
%function [states, probs] = get_bm_dist(weights, biases)
%
% Inputs:
%        weights nxn number of nodes, n (<20 as this is an exponential algorithm)
%         biases nx1 bias weights
%
% Outputs:
%         states Nxn All possible states, N = 2^n
%          probs Nx1 Normalized probs
%
% Elements of states are in {1,0}
% 
% log_prob(state) = const + sum_i state(i)*biases(i) + sum_{i<j} state(i)*state(j)*weights(i,j)
% Note that lower diagonal of coupling matrix is ignored.

% Iain Murray, August 2007

nn = length(biases);
if ~isequal(size(weights),[nn,nn])
    error('Sizes of weights and biases are not compatible');
end
if nn > 20
    error('Not running, might max out memory. Think more carefully. Maybe recode.');
end

% Remove bottom lower diagonal in case it contains junk
weights(find(tril(ones(size(weights))))) = 0;

num_states = 2^nn;
states = bitget(repmat((0:num_states-1)',1,nn),...
    repmat(1:nn, num_states,1)); % 2^nn x nn

bias_energy = states*biases(:);

coupling_energy = zeros(num_states,1);
for i = 1:nn
    coupling_energy = coupling_energy + ...
        states.*repmat(states(:,i),1,nn)*weights(:,i);
end

log_probs = bias_energy + coupling_energy;
log_Z = logsumexp(log_probs);
probs = exp(log_probs - log_Z);

