% Author: Jascha Sohl-Dickstein (2009)
% Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

function [f, df] = PL_RBM( W, Xin )
    
    ndims = size(Xin,1); % for the bias unit
    nbatch = size( Xin, 2 );

    E = E_RBM( W, Xin );
    Eflip = zeros( ndims, nbatch );    
    for i = 1:ndims
        X = Xin;
        X(i,:) = 1-X(i,:);
        Eflip(i,:) = E_RBM( W, X );
    end
    
    f = -E*ndims - sum( log( bsxfun( @plus, exp(-E), exp( -Eflip ) ) ), 1 );
    f = sum(f);
    
    df = 0;
    for i = 1:ndims
        X = Xin;
        X(i,:) = 1-X(i,:);
        rscl = sigmoid( E - Eflip(i,:) );
        %        rscl
        df = df + dE_RBM( W, X, rscl );
        df = df - dE_RBM( W, Xin, rscl );
        %[dE_RBM( W, X, rscl ), dE_RBM( W, Xin, rscl )]
    end
    
    df = df(:);
    
    f = -f / nbatch / ndims;
    df = -df / nbatch / ndims;
    
    %    if ~isfinite( f ) | sum( ~isfinite(df))>0
    %    f = 1e12;
    %    df(:) = 0;
    %end