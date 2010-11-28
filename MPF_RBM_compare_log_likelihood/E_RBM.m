% Author: Jascha Sohl-Dickstein (2009)
% Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function E = E_RBM( theta, X )

	 ndims = size(X, 1 );
	 nbatch = size(X, 2 );
    
         W = reshape( theta, prod(size(theta))/(ndims+1), ndims+1 );
         nhid = size(W,1)-1;
         W(nhid+1,ndims+1) = 0;
         Xf = ones( ndims+1,nbatch );
         Xf(1:ndims,:) = X;
                 
         ff = W*Xf;
         eff = exp(-ff);
         eff(nhid+1,:) = 0;
         E = -sum( log( 1 + eff ), 1 );
         E = E + ff(nhid+1,:);
