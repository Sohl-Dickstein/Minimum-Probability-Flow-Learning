% Author: Jascha Sohl-Dickstein (2009)
% Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function X_out = sample_ising( W, nsamples, burnin, independent_steps )
    nsamplingsteps = burnin + floor((nsamples-1)*(independent_steps));
	ndims = size(W, 1 );
 
    % choose dimensions to gibbs sample
    upd_i = floor( rand( nsamplingsteps, 1 ) * ndims ) + 1;
	%randi([0,ndims], nsamplingsteps, 1 );
    % precalculate the random nubers for comparison
    uni_rand = rand( nsamplingsteps, 1 );

    X_out = zeros( ndims, nsamples );
    x = floor( rand( ndims, 1 ) * 2 );
%    x = 2*floor( rand( ndims, 1 ) * 2 ) - 1;
%    x = 2*x - 1

    i_out = 1;
    next_sample = burnin;

%    print x

     bias = diag(W);
     W = W - diag(diag(W));

    for si = 1:nsamplingsteps
        E_act = 2*W(upd_i(si),:) * x + bias(upd_i(si));
        p_act = sigmoid(-E_act); % ****** sampling 0/1 here
        if p_act > uni_rand(si)
            x(upd_i(si)) = 1;
        else
            x(upd_i(si)) = 0;
	end	   
        if si == next_sample % copy to the output array if appropriate
            next_sample = si + independent_steps;
            X_out(:,i_out) = x;
            i_out = i_out + 1;
	end
    end    
