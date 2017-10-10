% CaImDeconvolution.m
%
%  Try algorithms to deconcolve calcium imaging data

% simulated data example
frameRate = 30; % calcium imaging acquisition frame rate (Hz)
delta = 1/frameRate;
firingRate = 0.75;
tau = 0.5; % 1 second decay constant
time = 0:100;
kernel = exp(-time./(tau*frameRate));

numFrames = 5000;noiseVariance = 0.2;
spikeTrain = poissrnd(firingRate*delta,[numFrames,1]);

noise = normrnd(0,sqrt(noiseVariance),[numFrames,1]);

temp = conv(spikeTrain,kernel);
F = temp(1:numFrames)+noise;

% normalize such that 0 is the 5-th percentile and 1 is the 80-th percentile
temp = quantile(F,[0.05,0.8]);

F = (F-temp(1))./(temp(2)-temp(1));

% attempt to implement Vogelstein's algorithm from 2010 ... Fast,
% non-negative deconvolution ...

% model parameter initializations
baseline = median(F);
sigmasquare = (1.4826*mad(F,1))^2;
precision = 1/(sigmasquare);
tau = 1; % 1 second approximate decay time for calcium
delta = 1/frameRate;
alpha = 1;
lambda = 1; % 1 Hz approximate firing rate, such that the poisson rate
        % in each bin is lambda*delta
       
 % C = smooth(F,10);
        
% try to solve with MCMC
  % F-t = alpha*C-t+baseline+error
  % n = Ct-gamma*C-(t-1)
% put priors on all the parameters ... what is the range of firing rates we
%   would expect?? ... conjugate prior to the poisson rate lambda is
%   the gamma distribution

gammaPrior_lambda = [1.4,5]; % check this distribution, fairly reasonable
                   % for expected firing rate of a neuron
normalPrior_baseline = [baseline,100];
gammaPrior_tau = [4,0.25];
gammaPrior_precision = [1e-3,1e3];
gammaPrior_alpha = [2,10];
Nparams = numFrames;
numParams = Nparams+5;

numIter = 3e5;
burnIn = 2e5;

skipRate = 200;

params = zeros(numParams,(numIter-burnIn)/skipRate);
posterior = zeros((numIter-burnIn)/skipRate,1);

params(1:Nparams,1) = poissrnd(lambda*delta,[Nparams,1]);
params(Nparams+1,1) = lambda; 
params(Nparams+2,1) = baseline;
params(Nparams+3,1) = tau; 
params(Nparams+4,1) = precision; 
params(Nparams+5,1) = alpha;

proposalSigma = zeros(numParams,1);
nSigma = sqrt(var(poissrnd(lambda*delta,[1500,1])));
for ii=1:Nparams
   proposalSigma(ii) = nSigma/2; 
end

proposalSigma(Nparams+1) = sqrt(var(gamrnd(gammaPrior_lambda(1),...
    gammaPrior_lambda(2),[1500,1])));
proposalSigma(Nparams+2) = 1;
proposalSigma(Nparams+3) = sqrt(var(gamrnd(gammaPrior_tau(1),...
    gammaPrior_tau(2),[1500,1])));
proposalSigma(Nparams+4) = 1;
proposalSigma(Nparams+5) = 1;

% make a surrogate beta-distributed random variable for each of the n-t
%  values, then use those to flip a coin on each iteration of MCMC, which
%  becomes the spike train, then use the spike train generated to calculate
%  the prior and the likelihood

nt = params(1:Nparams,1);

% nt = spikeTrain;

kernel = exp(-time./(params(Nparams+3,1)*frameRate));

C = conv(nt,kernel);C = C(1:numFrames);

% b = glmfit(C,F,'normal');
% params(Nparams+5,1) = b(2);params(Nparams+2,1) = b(1);

mu = F-params(Nparams+5,1).*C-params(Nparams+2,1);
loglikelihood = sum(log(normpdf(mu,0,sqrt(1/params(Nparams+4,1)))));

for ii=1:2e4
    tempAlpha = gamrnd(gammaPrior_alpha(1),gammaPrior_alpha(2));
    tempBaseline = normrnd(normalPrior_baseline(1),2);
    tempTau = gamrnd(gammaPrior_tau(1),gammaPrior_tau(2));
    tempLambda = gamrnd(gammaPrior_lambda(1),gammaPrior_lambda(2));
    tempSigma = 1/gamrnd(gammaPrior_precision(1),gammaPrior_precision(2));
    tempNt = poissrnd(tempLambda*delta,[Nparams,1]);
    
    kernel = exp(-time./(tempTau*frameRate));
    C = conv(tempNt,kernel);C = C(1:numFrames);
    
    mu = F-tempAlpha.*C-tempBaseline;
    tempLikely = sum(log(normpdf(mu,0,sqrt(tempSigma))));
    
    if tempLikely >= loglikelihood
        params(1:Nparams,1) = tempNt;
        params(Nparams+1,1) = tempLambda; 
        params(Nparams+2,1) = tempBaseline;
        params(Nparams+3,1) = tempTau; 
        params(Nparams+4,1) = 1/tempSigma; 
        params(Nparams+5,1) = tempAlpha;
        loglikelihood = tempLikely;
    end
end

logprior = sum(log(poisspdf(params(1:Nparams,1),delta*params(Nparams+1,1))))+...
    log(normpdf(params(Nparams+2,1),normalPrior_baseline(1),...
    normalPrior_baseline(2)))+log(gampdf(params(Nparams+3,1),...
    gammaPrior_tau(1),gammaPrior_tau(2)))+...
    log(gampdf(params(Nparams+4,1),gammaPrior_precision(1),...
    gammaPrior_precision(2)))+log(gampdf(params(Nparams+5,1),...
    gammaPrior_alpha(1),gammaPrior_alpha(2)))+...
    log(gampdf(params(Nparams+1,1),gammaPrior_lambda(1),gammaPrior_lambda(2)));

posterior(1) = loglikelihood+logprior;

proposalMu = zeros(numParams,1);

updateMu = zeros(numParams,1);
updateMu(1:Nparams,1) = poissrnd(lambda*delta,[Nparams,1]);
updateMu(Nparams+1,1) = lambda*1.5; % for mcmc, get (0,inf) to (-inf,inf)
updateMu(Nparams+2,1) = baseline/1.5;
updateMu(Nparams+3,1) = tau*0.8; % for mcmc, get (0,inf) to (-inf,inf)
updateMu(Nparams+4,1) = precision*2; % for mcmc, (0,inf) to (-inf,inf)
updateMu(Nparams+5,1) = alpha*0.9;
updateParam = 0.01;
loglambda = log(2.38^2);
optimalAccept = 0.234;logA = 0;
for ii=2:burnIn
    pStar = params(:,1)+normrnd(proposalMu,exp(loglambda).*proposalSigma);
    pStar(1:Nparams) = round(pStar(1:Nparams));
    
    nt = pStar(1:Nparams);
    
    if min(nt) >= 0 && sum(pStar([Nparams+1,Nparams+3,Nparams+4,Nparams+5])<=0)==0
        kernel = exp(-time./(pStar(Nparams+3,1)*frameRate));
        
        C = conv(nt,kernel);C = C(1:numFrames);
        
        mu = F-pStar(Nparams+5,1).*C-pStar(Nparams+2,1);
        loglikelihood = sum(log(normpdf(mu,0,sqrt(1/pStar(Nparams+4,1)))));
        
        logprior = sum(log(poisspdf(nt,delta*pStar(Nparams+1,1))))+...
            log(normpdf(pStar(Nparams+2,1),normalPrior_baseline(1),...
            normalPrior_baseline(2)))+log(gampdf(pStar(Nparams+3,1),...
            gammaPrior_tau(1),gammaPrior_tau(2)))+...
            log(gampdf(pStar(Nparams+4,1),gammaPrior_precision(1),...
            gammaPrior_precision(2)))+log(gampdf(pStar(Nparams+5,1),...
            gammaPrior_alpha(1),gammaPrior_alpha(2)))+...
            log(gampdf(pStar(Nparams+1,1),gammaPrior_lambda(1),gammaPrior_lambda(2)));
        
        tempPost = loglikelihood+logprior;
        logA = tempPost-posterior(1);
        
        if log(rand) < logA
            params(:,1) = pStar;
            posterior(1) = tempPost;
        end
%     end 
        loglambda = loglambda+updateParam.*(exp(min(0,logA))-optimalAccept);
    else
        loglambda = loglambda+updateParam*(-optimalAccept);
    end
    
    if mod(ii,400)==0
        meanSubtract = params(:,1)-updateMu;
        updateMu = updateMu+updateParam.*meanSubtract;
        proposalSigma = proposalSigma+updateParam.*(meanSubtract.*meanSubtract-...
            proposalSigma);
    end

   % scatter(ii,posterior(1));hold on;pause(0.01);
end

currentParams = params(:,1);
currentPost = posterior(1);
proposalSigma = exp(loglambda).*proposalSigma;

count = 2;
for ii=2:(numIter-burnIn)
    pStar = currentParams+normrnd(proposalMu,proposalSigma);
    pStar(1:Nparams) = round(pStar(1:Nparams));
    
    nt = pStar(1:Nparams);
    
    if min(nt) >= 0 && sum(pStar([Nparams+1,Nparams+3,Nparams+4,Nparams+5])<=0)==0
        kernel = exp(-time./(pStar(Nparams+3,1)*frameRate));
        
        C = conv(nt,kernel);C = C(1:numFrames);
        
        mu = F-exp(pStar(Nparams+5,1)).*C-pStar(Nparams+2,1);
        loglikelihood = sum(log(normpdf(mu,0,sqrt(1/pStar(Nparams+4,1)))));
        
        logprior = sum(log(poisspdf(nt,delta*pStar(Nparams+1,1))))+...
            log(normpdf(pStar(Nparams+2,1),normalPrior_baseline(1),...
            normalPrior_baseline(2)))+log(gampdf(pStar(Nparams+3,1),...
            gammaPrior_tau(1),gammaPrior_tau(2)))+...
            log(gampdf(pStar(Nparams+4,1),gammaPrior_precision(1),...
            gammaPrior_precision(2)))+log(gampdf(pStar(Nparams+5,1),...
            gammaPrior_alpha(1),gammaPrior_alpha(2)))+...
            log(gampdf(pStar(Nparams+1,1),gammaPrior_lambda(1),gammaPrior_lambda(2)));
        
        tempPost = loglikelihood+logprior;
        logA = tempPost-currentPost;
        
        if log(rand) < logA
            currentParams = pStar;
            currentPost = tempPost;
        end
    end
    if mod(ii,skipRate) == 0
        params(:,count) = currentParams;
        posterior(count) = currentPost;
        count = count+1;
    end
end

posteriorMean = mean(params,2);
nt = posteriorMean(1:Nparams);

kernel = exp(-time./(posteriorMean(Nparams+3,1)*frameRate));

C = conv(nt,kernel);C = C(1:numFrames);

mu = F-posteriorMean(Nparams+5,1).*C-posteriorMean(Nparams+2,1);

figure();
subplot(2,2,1);plot(nt);subplot(2,2,2);plot(C)
subplot(2,2,3);plot(spikeTrain);subplot(2,2,4);plot(F);

figure();
for ii=1:5
    subplot(5,1,ii);histogram(params(Nparams+ii,:));
end

[~,p] = corrcoef(F,C);
p(1,2)