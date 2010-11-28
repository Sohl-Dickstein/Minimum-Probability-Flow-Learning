% perform parameter estimation in a Restricted Boltzmann Machine using Contrastive Divergence

% Author: Jascha Sohl-Dickstein (2010)
% Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function W = train_CD( X0, W, k, weight_decay, steps, eta )
         
    d = size(X0,1);
    batch_size = size(X0,2);
    nhid = size(W,1)-1;
    nvis = size(W,2)-1;
    ndims = d;
    
    for t = 1:steps
        Xk = X0;
        x = ones( d + 1, batch_size );
        x(1:ndims,:) = Xk;
        % do k gibbs sampling steps
        for ii = 1:k
            % sample up
            xhid = sigmoid( -W*x );
            xhid = (xhid > rand(size(xhid)));
            xhid(nhid+1,:) = 1; % bias unit
            % sample down
            x = sigmoid( -W'*xhid );
            x = (x > rand(size(x)));
            x(ndims+1,:) = 1; % bias unit
        end
        Xk = x(1:d,:);
            
        dE0 = dE_RBM( W, X0, 1/batch_size );
        dEk = dE_RBM( W, Xk, 1/batch_size );
        W(:) = W(:) + eta*(- dE0(:) + dEk(:));
        
        % weight decay!
        W = W - weight_decay*eta*W;
    end
