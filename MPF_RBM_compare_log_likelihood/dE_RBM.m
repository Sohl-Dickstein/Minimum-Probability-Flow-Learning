% Author: Jascha Sohl-Dickstein (2009)
% Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function dE = dE_RBM( theta, X, rscl )

    	 ndims = size(X, 1 );
	 nbatch = size(X, 2 );

         W = reshape( theta, prod(size(theta))/(ndims+1), ndims+1 );
         nhid = size(W,1)-1;
         W(nhid+1,ndims+1) = 0;
         Xf = ones( ndims+1,nbatch );
         Xf(1:ndims,:) = X;

         eff = exp(W*Xf);

         ileff = 1./(1 + eff );
         ileff(nhid+1,:) = Xf(ndims+1,:);

         %         dE = ileff * diag(rscl) * Xf';
         dE = ileff * bsxfun( @times, rscl(:)', Xf )';

         dE(nhid+1, ndims+1) = 0;
         
         dE = dE(:);
