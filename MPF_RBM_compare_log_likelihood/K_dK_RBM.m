% MPF objective function and gradient for RBM, with transitions allowed between all states differing by a single bit flip.  See MPF_RBM_objective.pdf.

% Author: Jascha Sohl-Dickstein (2009)
% Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function [K, dK] = K_dK_RBM( theta, Xin )

    ndims = size(Xin,1); % for the bias unit
    nbatch = size( Xin, 2 );
    nparms = size( theta, 1 );
    nexperts = nparms / (ndims+1) - 1; % +1 and -1 for bias
    J = reshape( theta, [nexperts+1, ndims+1] );
    
    % the last column of J holds the bias term for the hidden units
    % the last row of J holds the bias term for the visible units    
    
    bias = J(end,:); % bias for visible units
    bias(end) = 0;
    J = J(1:end-1,:);
    
    X = ones( ndims+1, nbatch ); % last dimension is bias for hidden units
    X(1:end-1,:) = Xin;
    
    ff = J * X; % feedforward - this should be the most expensive step!
    expnegff = exp( -ff );
    expnegffp = 1 + expnegff;
    Ekj = -log( expnegffp );
    Ek = sum( Ekj, 1 );
%    bX = bias * X;
%    Ek = Ek + bX;     % visible bias units
    expr = expnegff ./ expnegffp;
    
    K = 0;
    dKdbias = zeros(1, ndims+1 );
    dKdJ = zeros( nexperts, ndims+1 );
    
    os = ones(nexperts, 1);
    ot = ones(1, nbatch);
    
    p1 = 0;
    
    for d = 1:ndims
        % the energies of all the non-data states with bit d flipped
%        ffd = ff + ( J(:,d) * (1-2*X(d,:)) );
%        expnegffd = exp( -ffd );
        epj = exp(J(:,d));
        enj = 1./epj;
%        expnegffd = expnegff .* (exp(J(:,d)) * X(d,:) + exp(-J(:,d)) * (1-X(d,:)));
        expnegffd = expnegff .* ((epj - enj) * X(d,:) + enj * ot);
        %Ekjd = log( 1 + expnegffd );
        %Ekd = sum( Ekjd, 1 );
        expnegffdp = expnegffd + 1;
        
        expnegffdpr = expnegffdp ./ expnegffp;
        Ekjdr = prod( expnegffdpr, 1 );
        Ekdr = log( Ekjdr );
        Ekdr = Ekdr - bias(d) * (1-2*X(d,:));     % visible bias
        expdiff = exp( 0.5 * Ekdr );       
%        Ekjd = prod( expnegffdp, 1 );
%        Ekd = -log( Ekjd );
%        Ekd = Ekd + bias(d) * (1-2*X(d,:));     % visible bias
%        
%        expdiff = exp( 0.5 * ( Ek - Ekd ) );
        K = K + sum( expdiff );
        
        if ~isfinite(K)
            blar = 1;
        end
        
        exprd = expnegffd ./ (expnegffdp);
        %op = (-expr + exprd) * diag(expdiff) * X';
        %op(:,d) = op(:,d) + 2 * exprd * diag(expdiff) * X(d,:)';
%        op = (-expr + exprd) * (expdiff( ones(ndims,1),:) .* X)';
%        p1( :, d, : ) = (-expr + exprd);
%        p2( d, :, : ) = (expdiff( ones(ndims,1),:) .* X)';
%        p2( d, :, : ) = (expdiff' * os) .* Xt;
%        op = (-expr + exprd) * Xt;
%        dK = dK + (-expr + exprd) * ((expdiff' * os) .* Xt);
        p1 = p1 + (expr - exprd) .* (os * expdiff);
        dKdJ(:,d) = dKdJ(:,d) - exprd * (expdiff .* (1 - 2*X(d,:)))';
        
        dKdbias(d) = -0.5 * expdiff * (1-2*X(d,:))';
%        dK = dK + op/2;

%        disp(d)
        
        %        K = sum(Ekd);
        %        dK = -exprd * X';
        %        dK(:,d) = dK(:,d) - exprd * (1 - 2 * X(d,:)');

    end
    
    dKdJ = dKdJ + p1 * X';
    
    dK = zeros( nexperts+1, ndims+1 );
    dK(1:nexperts, : ) = dKdJ / 2;
    dK(end,:) = dKdbias;
    
%    p1 = reshape(p1, [nexperts, (ndims-1)*nbatch] );
%    p2 = reshape(p2, [(ndims-1)*nbatch, ndims] );
%    dK = dK + p1 * p2 / 2;
    
    %    K = sum(Ek);
    %    dK = -expr * X';
    
    K  = K  / nbatch;
    dK = dK / nbatch;

    
    dK = dK(:);
end