% computes log likelihood of data X under an RBM described by W

% Author: Jascha Sohl-Dickstein (2009)
% Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function L = compute_log_likelihood( X, W )
    
% generate a list of all possible states
d = size(X, 1 );
N = 2^d;
binary_lookup = zeros( N, d );
for i = 2:N
    binary_lookup( i,: ) = binary_lookup( i-1,: );
    binary_lookup( i,1 ) = binary_lookup( i,1 ) + 1;
    for j = 1:d-1
        if binary_lookup( i, j ) == 2
            binary_lookup( i, j )   = 0;
            binary_lookup( i, j+1 ) = binary_lookup( i, j+1 ) + 1;
        end
    end
end
Xall = binary_lookup';

%whos Xall

% calculate the energy for each of them
Eall = E_RBM( W, Xall );
% and the partition function
Z = sum( exp( -Eall ) );
logZ = log(Z);

% calculate the energy for the data distribution
E = E_RBM( W, X );

L = -mean(E) - logZ;

L = L / log(2);

mean(E) / log(2)
logZ / log(2)
