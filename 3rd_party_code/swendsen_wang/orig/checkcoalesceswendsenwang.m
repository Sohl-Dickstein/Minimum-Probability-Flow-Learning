function S = checkcoalesceswendsenwang(J,observed,labels,burn,iters)
% function S = checkcoalesceswendsenwang(J,observed,labels,burn,iters)
% Generalised Swendsen-Wang sampler for Ising models (spins of +/-1) with
% positive and negative connections. Makes sure to use random numbers in
% careful way to encourage chains using same seed but different initial
% conditions to coalesce. Can then check for this with a few initial conditions
% to see if CFTP seems feasable, before I spend ages coding it.
%
% REMEMBER to use {+1,-1} spins (NOT {0,1}) or this will break horribly.
% 
% INPUTS
%        J nxn --- coupling matrix
% observed nx1 --- mask giving variables that are observed (!=0 => observed)
%   labels nx1 --- labels for observed variables (non-observeds are used as
%                  start state)
%     burn 1x1 --- number of iterations to discard
%    iters 1x1 --- number of iterations for which to run after burn steps
%
% OUTPUTS
%        S nxiters --- samples over labels
%
% Iain Murray --- March 2004

n=size(J,1);
S=zeros(n,iters);
D=zeros(n,n);

% Split up weights into positive and negative parts
Jpos=J.*(J>0); Jneg=J.*(J<0);

% Use labels provided by user for initial condition
Y=labels;

for s=1:(burn+iters)
	% Sample from the links between clusters
	Ymatch=(repmat(Y,1,n)==repmat(Y',n,1));
	Yunmatch=(repmat(Y,1,n)~=repmat(Y',n,1));
	U=rand(n,n);
	D=tril((U<(1-exp(-2*Jpos))).*Ymatch,-1);
	E=tril((U<(1-exp(2*Jneg))).*Yunmatch,-1);
	D=D+D'+eye(n); E=E+E'; F=D+E;
	
	% Sample from the variables given the links
	Q=2*(rand(n,n)<0.5)-1;
	curpos=randperm(n); % gives position in ordering for vars
	                    % eg curpos(5) gives postion of variable 5
	sampled=zeros(n,1);
	%clustersizes=[];
	for i=1:n
		if ~sampled(i)
			% find all of connected component owning this point
			cluster=i;
			newcluster=find(F(i,:));
			while (~isequal(cluster,newcluster))
				cluster=newcluster;
				newcluster=find(sum(F(:,cluster),2));
			end
			%clustersizes=[clustersizes,max(size(cluster))];
			% Mark every point in this cluster done
			sampled(cluster)=1;
			% If we have label in cluster that dictates everything
			% about it so we leave it. Otherwise we flip all lables
			% with probability 1/2.
			existing=find(observed.*sum(F(:,cluster),2));
			if min(size(existing))==0
				% Take on colour suggested by node highest in
				% the current random ordering
				[dummy,idx]=max(curpos'.*sum(F(:,cluster),2));
				if (Y(idx)~=Q(idx))
					Y(cluster)=-Y(cluster);
				end
			end
		end
	end
%	clustersizes=sort(clustersizes);
%	clustersizes=clustersizes(end:-1:end-10)
%	hist(clustersizes), drawnow

	% Store sample
	if s>burn
		S(:,s-burn)=Y;
	end
end
