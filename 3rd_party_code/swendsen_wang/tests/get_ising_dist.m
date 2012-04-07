function [states, probs] = get_ising_dist(ising_J, ising_h)
%function [states, probs] = get_ising_dist(ising_J, ising_h)
%
% Inputs:
%        ising_J nxn number of nodes, n (<20 as this is an exponential algorithm)
%        ising_h nx1 field parameters (bias weights)
%
% Outputs:
%         states Nxn All possible states, N = 2^n
%          probs Nx1 Normalized probs
%
% Elements of states are in {-1, +1}
% 
% log_prob(state) = const + sum_i state(i)*ising_h(i) + sum_{i<j} state(i)*state(j)*ising_J(i,j)
% Note that lower diagonal of coupling matrix is ignored.

% Iain Murray, August 2007

nn = length(ising_h);
if ~isequal(size(ising_J),[nn,nn])
    error('Sizes of ising_J and ising_h are not compatible');
end
if nn > 20
    error('Not running, might max out memory. Think more carefully. Maybe recode.');
end

% Remove bottom lower diagonal in case it contains junk
ising_J(find(tril(ones(size(ising_J))))) = 0;

num_states = 2^nn;
states = 2*bitget(repmat((0:num_states-1)',1,nn),...
    repmat(1:nn, num_states,1)) - 1; % 2^nn x nn

field_energy = states*ising_h(:);

coupling_energy = zeros(num_states,1);
for i = 1:nn
    coupling_energy = coupling_energy + ...
        states.*repmat(states(:,i),1,nn)*ising_J(:,i);
end

log_probs = field_energy + coupling_energy;
log_Z = logsumexp(log_probs);
probs = exp(log_probs - log_Z);

