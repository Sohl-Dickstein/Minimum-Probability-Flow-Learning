% - generates a weight matrix for an RBM, and then generates samples from that RBM
% - using the generated sample, trains RBMs using Minimum Probability Flow learning, pseudolikelihood, CD-1, and CD-10
% - compares the log likelihood of the samples for each of the training techniques

% Author: Jascha Sohl-Dickstein (2010)
% Web: http://redwood.berkeley.edu/wiki/Jascha_Sohl-Dickstein
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)

% make experiments repeatable
rng(0);

d_vis = 10; % number of units in the visible layer
d_hid = 10; % number of units in the hidden layer
batch_size = 100; % number of training samples to generate

CD_steps = 10000; % number of CD learning steps to do
CD_eta = 0.01; % eta to use for stochastic gradient descent for CD

fprintf( 'choosing random RBM\n' );
tic();
Wtrue = 4*randn( d_hid+1, d_vis+1 ) / sqrt(d_vis+1);
independent_steps = 200 % how many steps to go between samples
fprintf( 'generating data samples\n' );
X = sample_RBM( Wtrue, batch_size, independent_steps, independent_steps, rand( d_vis, 1 ) > 0.5 );
X = X > rand(size(X));
fprintf( 'computing log likelihood of true model\n' );
L_true = compute_log_likelihood( X, Wtrue )
toc()



tic();
fprintf( 'initializing training weights\n' );
Winit = randn( d_hid+1, d_vis+1 ) / sqrt(d_vis+1);
fprintf( 'computing log likelihood\n' );
L_init = compute_log_likelihood( X, Winit )
toc()

tic()
% pseudolikelihood
fprintf( 'estimating weight matrix via pseudolikelihood\n' );
Wpl = Winit;
minf_options = [];
minf_options.maxlinesearch = 5000;
[Wplfn] = minFunc( @PL_RBM, Wpl(:), minf_options, X );
Wpl(:) = Wplfn(:);
toc()
tic()
fprintf( 'computing log likelihood\n' );
L_pl = compute_log_likelihood( X, Wpl )
toc()


tic();
fprintf( 'estimating weight matrix via MPF\n' );
% MPF
Wmpf = Winit;
minf_options = [];
minf_options.maxlinesearch = 5000;
Wmpfn = minFunc( @K_dK_RBM, Wmpf(:), minf_options, X );
Wmpf(:) = Wmpfn(:);
toc()
tic()
fprintf( 'computing log likelihood\n' );
L_mpf = compute_log_likelihood( X, Wmpf )
toc()


tic();
Wcd1 = Winit;
Wcd10 = Winit;
Wcd1wd = Winit;
Wcd10wd = Winit;

% CD-1
fprintf( 'estimating weight matrix via CD1\n' );
Wcd1 = train_CD( X, Wcd1, 1, 0, CD_steps, CD_eta );
toc()
tic()
fprintf( 'computing log likelihood\n' );
L_cd1 = compute_log_likelihood( X, Wcd1 )
% CD-10
toc()
tic()
fprintf( 'estimating weight matrix via CD10\n' );
Wcd10 = train_CD( X, Wcd10, 10, 0, CD_steps, CD_eta );
toc()
tic()
fprintf( 'computing log likelihood\n' );
L_cd10 = compute_log_likelihood( X, Wcd10 )
toc()

tic()
% CD-1 weight decay
fprintf( 'estimating weight matrix via CD1 with weight decay\n' );
wd = 0.01;
wd = 0.1;
Wcd1wd = train_CD( X, Wcd1wd, 1, wd, CD_steps, CD_eta );
toc()
tic()
fprintf( 'computing log likelihood\n' );
L_cd1wd = compute_log_likelihood( X, Wcd1wd )
% CD-10 weight decay
toc()
tic()
fprintf( 'estimating weight matrix via CD10 with weight decay\n' );
Wcd10wd = train_CD( X, Wcd10wd, 10, wd, CD_steps, CD_eta );
toc()
tic()
fprintf( 'computing log likelihood\n' );
L_cd10wd = compute_log_likelihood( X, Wcd10wd )
toc()

fprintf( '\nLog likelihoods (more positive is better):\ninitialized\t %f\nMPF \t\t %f\nCD1 \t\t %f\nCD10 \t\t %f\nCD1 wd\t\t %f\nCD10 wd\t\t %f\npseudolikelihood %f\ntrue params\t %f\n\n', L_init, L_mpf, L_cd1, L_cd10, L_cd1wd, L_cd10wd, L_pl, L_true );
