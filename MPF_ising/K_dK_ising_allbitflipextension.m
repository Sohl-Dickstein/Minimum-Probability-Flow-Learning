% this function implements the minimum probability flow objective function
% and gradient for a fully connected Ising model.  see
% http://arxiv.org/abs/0906.4779
%
% This code differs from K_dK_ising.m in that every state is compared to
% one more state than in K_dK_ising.m.  This additional state is the
% all-bits-flipped state.  This additional comparison significantly
% improves performance in cases (such as neural spike codes) where
% activation is extremely sparse, because it allows the MPF algorithm to
% correctly weight states where many units are turned on.
%
% this code is written under the greatly accelerating assumption that
% samples in the data vector differ from each other by more than one bit
% flip (see http://arxiv.org/abs/0906.4779).  this is nearly always true
% for large systems, but means you'll get funny answers if you do a test
% run with a system of only a few units with lots of data.
%
% see MPF_Ising_objective.pdf for a derivation of this objective
%
% Author: Jascha Sohl-Dickstein (2012)
% Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function [K, dK] = K_dK_ising_allbitflipextension( J, X )
    [ndims, nbatch] = size( X );

    J = reshape( J, [ndims, ndims] );
    J = (J + J')/2;

    Y = J*X;
    diagJ = diag(J);
    % XnotX contains (X - [bit flipped X])
    XnotX = 2*X-1;
    % Kfull is a [ndims, nbatch] matrix containing the contribution to the objective function from flipping each bit in the rows, for each datapoint on the columns
    Kfull = exp(XnotX .* Y - (1/2)*diagJ(:,ones(1,nbatch))); 
    K = sum(Kfull(:));
    
    lt = Kfull.*XnotX;
    dJ = lt * X';
    dJ = dJ - (1/2)*diag( sum(Kfull, 2) );
             
    
    %% add all bit flip comparison case
    % calculate the energies for the data states
    EX = sum( X.*Y );
    % calculate the energies for the states where all bits are flipped relative to data states
    notX = 1-X;
    notY = J*notX;
    EnotX = sum( notX.*notY );
    % calculate the contribution to the MPF objective function from all-bit-flipped states
    K2full = exp( (EX - EnotX)/2 );
    K2 = sum(K2full);
    % calculate the gradient contribution from all-bit-flipped states
    dJ2 = bsxfun( @times, X, K2full ) * X'/2 - bsxfun( @times, notX, K2full ) * notX'/2;
    % add all-bit-flipped contributions on to full objective
    K = K + K2;
    dJ = dJ + dJ2;
    
    
    % symmetrize coupling matrix
    dJ = (dJ + dJ')/2;
    dK = dJ(:);

    % average over batch
    K  = K  / nbatch;
    dK = dK / nbatch;
