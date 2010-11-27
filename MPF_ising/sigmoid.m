%
% Implements a Sigmoid (logisitic) function
%
 
function g = sigmoid (x)

g = 1 ./ (1 + exp(-x));