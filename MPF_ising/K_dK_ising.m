% this function implements the minimum probability flow objective function and gradient for a fully connected Ising model.  see http://arxiv.org/abs/0906.4779

% this code is written under the greatly accelerating assumption that samples in the data vector differ from each other by more than one bit flip (see http://arxiv.org/abs/0906.4779).  this is nearly always true for large systems, but means you'll get funny answers if you do a test run with a system of only a few units with lots of data.

% for the ising model case, calculating the energy change resulting from flipping a single bit only requires looking at a single column/row of the coupling matrix J (see jpegs of board for derivation - soon to TeX-ed).  this means that with some matrix operations and dense matlab code, the MPF objective function can be greatly accelerated. the code below is optimized to do this ... at the cost of making it tougher to read ...

% Author: Jascha Sohl-Dickstein (2009)
% Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function [K, dK] = K_dK_ising( J, X )
    [ndims, nbatch] = size( X );

    J = reshape( J, [ndims, ndims] );
    J = (J + J')/2;

    Y = J*X;
    diagJ = diag(J); 
    XnotX = 2*X-1;
    Kfull = exp(XnotX .* Y - (1/2)*diagJ(:,ones(1,nbatch)));
    K = sum(Kfull(:));
    
    lt = Kfull.*XnotX;
    dJ = lt * X';
    dJ = dJ - (1/2)*diag( sum(Kfull, 2) );
         
    dJ = (dJ + dJ')/2;
    dK = dJ(:);
    
    K  = K  / nbatch;
    dK = dK / nbatch;