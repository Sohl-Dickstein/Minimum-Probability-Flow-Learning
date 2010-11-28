% draw samples from an RBM
% this outputs probabilities of activations, not activations!!!

% Author: Jascha Sohl-Dickstein (2009)
% Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function X_out = sample_RBM( W, nsamples, burnin, independent_steps, x )

    % bias terms in last row and column of weight matrix

	nsamplingsteps = burnin + 1 + (nsamples-1)*(independent_steps);
	ndims = size(W, 2 )-1; % -1 because of bias terms
    nhid = size(W,1)-1;

    X_out = zeros( ndims, nsamples );
    %x = floor( rand( ndims, 1 ) * 2 );
    xt = x;
    x = ones(ndims+1,1);
    x(1:ndims) = xt;
    
    i_out = 1;
    next_sample = burnin+1;

    for si = 1:nsamplingsteps
        x(ndims+1) = 1; % bias unit
        % sample hidden
        xhid = sigmoid( -W*x );
        xhid = (xhid > rand(size(xhid)));
        xhid(nhid+1) = 1; % bias unit
        % resample visible
        x = sigmoid( -W'*xhid );
        
        if si == next_sample % copy to the output array if appropriate
            next_sample = si + independent_steps;
            X_out(:,i_out) = x(1:ndims);
            i_out = i_out + 1;
        end
        x = (x > rand(size(x)));
        %                sadasd
    end    

