function E = E_ising( theta, X )

    % ising model energy function

    ndims = size(X, 1 );
    nbatch = size(X, 2 );
    W = reshape( theta, ndims, ndims );
    % bias terms are on the diagonal of W
	 
    E = sum( (W*X).*X, 1 );
