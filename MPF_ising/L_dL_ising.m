% this function calculates the average negative log likelihood and gradient for a fully connected Ising model.  Note that this takes an amount of time exponential in the number of units, so use with caution with larger networks.  This can be substituted in place for K_dK_ising

% Author: Jascha Sohl-Dickstein (2012)
% Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function [L, dL] = L_dL_ising( J, X )
    [ndims, nbatch] = size( X );

    J = reshape( J, [ndims, ndims] );
    J = (J + J')/2;

    % get the energy for all the data states
    E = sum( X.*(J*X) );
    
    % fill in all possible binary patterns
    X_all = zeros( ndims, 2^ndims );
    for d = 1:ndims
        X_all(d,:) = bitget( 0:2^ndims-1, d );
    end

    % get the energy for all possible states
    E_all = sum( X_all.*(J*X_all) );
    E_offset = -min(E_all); % this is to keep the exponential below in a reasonable range -- if the weight matrix is large, then the energy can become very negative, and the exponential of the negative energy can exceed the range allowed by floating point.  so we add (and then later subtract) a constant so that the smallest energy is 0
    potential_all = exp( -(E_all+E_offset) ); % caclulate the potential function for all patterns    
    Z = sum( potential_all ); % calculate the partition function (actually, this is Z*exp(E_offset) )

    L = -sum(E)/nbatch - log( Z )-E_offset; % average log likelihood
    dL   = -X*X'/nbatch + 1./Z * bsxfun( @times, X_all, potential_all ) * X_all';
    dL = dL(:);

    % NEGATIVE log likelihood
    L = -L;
    dL = -dL;
