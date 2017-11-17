% NetworkInference.m
%
%  Take deconvolved calcium traces from calcium imaging and infer the 
%   undirected connectivity of a network of neurons

% load('spikeTrain.mat','S');
% 
% [T,N] = size(S);

load('SpikeDataExample.mat');

H = history;
[N,historyParams] = size(history);
numBases = 1;
B = normrnd(0,1,[historyParams,numBases]);

S = cov(Y);
d = size(Y,2);

iter = 1;
maxIter = 1e4;
tolerance = 1e-6;
h = 1e-3;

sigmasquare = var(Y(:));
W = H*B;
C = var(W(:))+sigmasquare.*eye(d); % W*W'

loglikelihood = -N/2*(d*log(2*pi)+log(det(C))+trace(pinv(C)*S));

lineSteps = [h,1e-4,1e-2,1e-1,0.5e-1,1,10];lineN = length(lineSteps);
lineLikelihood = zeros(lineN,1);

difference = 1;
parameters = [B(:);sigmasquare];
numParameters = length(parameters);
gradient = zeros(numParameters,1);
plot(iter,loglikelihood,'.');hold on;
while iter<maxIter || difference>tolerance
    for jj=1:numParameters
        tempParams = parameters;
        tempParams(jj) = tempParams(jj)+h;
        B = reshape(tempParams(1:end-1),[historyParams,numBases]);
        sigmasquare = tempParams(end);
        W = H*B;
        C = var(W(:))+sigmasquare.*eye(d);
        tempLikelihood = -N/2*(d*log(2*pi)+log(det(C))+trace(pinv(C)*S));
        gradient(jj) = (tempLikelihood-loglikelihood)/h;
    end
    
    saveParameters = parameters;
    for jj=1:numParameters
        lineLikelihood(1) = gradient(jj)*h+loglikelihood;
        for kk=2:lineN
            tempParams = parameters;
            tempParams(jj) = tempParams(jj)+sign(gradient(jj))*lineSteps(kk);
            B = reshape(tempParams(1:end-1),[historyParams,numBases]);
            sigmasquare = tempParams(end);
            W = H*B;
            C = var(W(:))+sigmasquare.*eye(d);
            lineLikelihood(kk) = -N/2*(d*log(2*pi)+log(det(C))+trace(pinv(C)*S));
        end
        [maxLikely,ind] = max(lineLikelihood);
        if maxLikely > loglikelihood
            saveParameters(jj) = parameters(jj)+sign(gradient(jj))*lineSteps(ind);
        else
            saveParameters(jj) = parameters(jj);
        end
    end
    B = reshape(saveParameters(1:end-1),[historyParams,numBases]);
    sigmasquare = saveParameters(end);
    W = H*B;
    C = var(W(:))+sigmasquare.*eye(d);
    tempLikelihood = -N/2*(d*log(2*pi)+log(det(C))+trace(pinv(C)*S));
    difference = tempLikelihood-loglikelihood;
    if difference>0
         parameters = saveParameters;
         loglikelihood = tempLikelihood;
    else
        break;  
    end
    iter = iter+1;
    plot(iter,loglikelihood,'.');pause(0.1);
end
B = reshape(parameters(1:end-1),[historyParams,numBases]);
sigmasquare = parameters(end);

[beta,dev,stats] = glmfit(H*B,Y,'normal');

AIC = dev+2*(numBases+1);

[~,dev,~] = glmfit(ones(N,1),Y,'normal','constant','off');

AIC1 = dev+2;
fprintf('Constant AIC: %3.2e\n',AIC1);
fprintf('Model AIC: %3.2e\n',AIC);