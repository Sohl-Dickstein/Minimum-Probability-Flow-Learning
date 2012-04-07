addpath('..');

nn = 4;

% Parameters
rand('state',1);
randn('state',1);
if 1
    weights = 0.9*randn(nn)/nn;
    weights = triu(weights, 1); % Get rid of lower diagonal
    biases = -rand(nn,1)/nn;
    weights
    biases
else
    weights = [0 -1;...
               0  0]
    biases = [0.3; 1.5]
end

% Ising representation for use in Swedsen-Wang algorithm
ising_J = (weights+weights')/4;
ising_h = biases/2 + sum(ising_J, 2);

% Initialize S-W sampler
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ising_state = (rand(nn,1) < 0.5)*2 - 1;
cum_state = zeros(size(ising_state));
cum_prob = zeros(size(ising_state));

% GO!
iters = 10000;
for i=1:iters
    [ising_state, cprobs] = sw_allall_ising(ising_state, ising_J, ising_h, 1);
    cum_state = cum_state + (ising_state==1);
    cum_prob = cum_prob + cprobs;
end

% Notice the second column is usually quite a bit better!
sw_marginal_ests = [cum_state, cum_prob]/iters


% Initialize BM Gibbs sampler
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
state = (rand(nn,1) < 0.5);
cum_state = zeros(size(ising_state));
cum_prob = zeros(size(ising_state));

% GO!
rand('state',sum(100*clock))
iters = 10000;
for i=1:iters
    [state, cprobs] = gibbs_allall_bm(state, weights, biases, 1);
    cum_state = cum_state + (state==1);
    cum_prob = cum_prob + cprobs;
end

% Notice the second column is usually quite a bit better!
gibbs_marginal_ests = [cum_state, cum_prob]/iters


% Initialize BM S-W sampler (not native-BM, uses Ising conversion internally)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
state = (rand(nn,1) < 0.5);
cum_state = zeros(size(ising_state));
cum_prob = zeros(size(ising_state));

% GO!
iters = 10000;
for i=1:iters
    [state, cprobs] = sw_allall_bm(state, weights, biases, 1);
    cum_state = cum_state + (state==1);
    cum_prob = cum_prob + cprobs;
end

% Notice the second column is usually quite a bit better!
sw2_marginal_ests = [cum_state, cum_prob]/iters


% Ground truth via BM representation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
[states, probs] = get_bm_dist(weights, biases);
true_marginal = zeros(size(sw_marginal_ests,1),1);
for i = 1:length(true_marginal)
    true_marginal(i) = sum(probs(find(states(:,i)==1)));
end
true_marginal


% Ground truth via Ising representation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
[states, probs] = get_ising_dist(ising_J, ising_h);
true_marginal2 = zeros(size(true_marginal));
for i = 1:length(true_marginal)
    true_marginal2(i) = sum(probs(find(states(:,i)==1)));
end

if max(abs(true_marginal-true_marginal2)) > 1e-9
    error('Ising and BM representations are not equivalent')
end

