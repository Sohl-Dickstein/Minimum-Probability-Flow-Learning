function S = swendsenwang(J,observed,labels,burn,iters)
% function S = swendsenwang(J,observed,labels,burn,iters)
% Generalised Swendsen-Wang sampler for Ising models (spins of +/-1) with
% positive and negative connections.
%
% REMEMBER to use +1/-1 spins throughout or weirdness results
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

% Use labels (whether clamped or not) provided by user as starting point.
Y=labels;

for s=1:(burn+iters)
	% Sample from the links between clusters
	Ymatch=(repmat(Y,1,n)==repmat(Y',n,1));
	Yunmatch=(repmat(Y,1,n)~=repmat(Y',n,1));
	D=tril((rand(n,n)<(1-exp(-2*Jpos))).*Ymatch,-1);
	E=tril((rand(n,n)<(1-exp(2*Jneg))).*Yunmatch,-1);
	D=D+D'+eye(n); E=E+E'; F=D+E;

	% Sample from the variables given the links
	sampled=zeros(n,1);
	for i=1:n
		if ~sampled(i)
			% find all of connected component owning this point
			cluster=i;
			newcluster=find(F(i,:));
			while (~isequal(cluster,newcluster))
				cluster=newcluster;
				newcluster=find(sum(F(:,cluster),2));
			end
			% Mark every point in this cluster done
			sampled(cluster)=1;
			% If we have label in cluster that dictates everything
			% about it so we leave it. Otherwise we flip all lables
			% with probability 1/2.
			existing=find(observed.*sum(F(:,cluster),2));
			if min(size(existing))==0
				if (rand<0.5)
					Y(cluster)=-Y(cluster);
				end
			end
		end
	end

	% Store sample
	if s>burn
		S(:,s-burn)=Y;
	end
end
