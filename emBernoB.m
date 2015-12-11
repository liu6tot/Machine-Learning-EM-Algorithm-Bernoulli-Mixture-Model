function [like,ll,new_miu,new_pi]=emBernoB(dataset,K)
%
% Input
% dataset = training data as D dimensional row vectors
% K = number of bernoulli fit
%
% Output
% like = vector of log likelihoods at each iteration
% ll = the 
%
[N,M]=size(dataset);
like=[];
thresh=1e-6;
converged=0;
iter=0;
ll=-inf;
prev=0;
maxiter=10000;

%initial pi and miu
ini_pi=rand(K,1);
ini_miu=rand(K,1);


while (iter<maxiter) && ~converged
  prev=ll;
  ll=0;
  
  if (iter ~= 0)
    ini_miu = new_miu;
  end
  
  %E-step 
  tau = zeros(N,K);
    for k = 1:K
        for n=1:N
          tau(n,k) = ini_pi(k,1) * ini_miu(k,1) ^ sum(dataset(n,:)) * (1-ini_miu(k,1))^(M - sum(dataset(n,:)));
        end
    end
    
  %loglikelihood
  for n=1:N
      %log
      l=sum(tau(n,:));
      ll=ll+log(l);
      %tau final
      tau(n,:)=tau(n,:)/l;
  end
  
  %convergence
  if (ll-prev<thresh)
    converged=1;
  end
 
  like=[like ll];
 
  % M-step 
   
    for k = 1:K
        %initial miu and pi 
        new_miu(k,:)=0;
        new_pi(k,:)=0;
        
        %calculating new pi 
        sumtau=sum(tau(:,k));
        new_pi(k,:) = sumtau/N; 
               
        for n = 1:N
            n_miu= sum(dataset(n,:))/M;
            tau_n=tau(n,k) * n_miu;
            new_miu(k,:) = new_miu(k,:)+ tau_n;
        end 
        new_miu(k,:) = new_miu(k,:)/ sumtau; %miu
        
    end 
    
 
  iter=iter+1; %iteration
  disp(iter)
  disp(new_miu)
  disp(new_pi)

end

    
    
    
    
