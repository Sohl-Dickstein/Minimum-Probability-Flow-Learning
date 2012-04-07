function [state, cprobs] = gibbs_allall_bm(state, weights, biases, iters)
% weights should only occupy upper triangle.
% The energy includes: 0.5 \sum_i \sum_j W_{ij} s_i s_j
% often written: \sum_{(i,j)} W_{ij} s_i s_j
% state should be {0,1} variables

weights(find(tril(ones(size(weights))))) = 0;
weights = (weights+weights');
num_cases = size(state, 2);

nn = length(biases);
ordering = randperm(nn);
cprobs = zeros(size(state));

for it = 1:(iters+1)
    for i = ordering
        c_activation = weights(i,:)*state + biases(i,:);
        cprobs(i,:) = 1./(1+exp(-c_activation));
        if state <= iters
            state(i,:) = sample_cond(cprobs(i,:));
        end
    end
end

