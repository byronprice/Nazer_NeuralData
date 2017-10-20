function [lambda,gama,est_spike] = L0_CV(ca_trace,Fs,tau_guess)
% L0_CV.m
%
%  Cross-validiation procedure for the L0 spike inference algorithm, 
%   for automatic estimation of the lambda and gamma free parameters, and
%   therefore a "best" inferred spike train
% INPUT:
%       ca_trace - an experimental calcium trace, preferrably one that has been 
%         cleaned up using some kind of robust regression (deltaF / F) ...
%       Fs - sampling frequency of the calcium imaging system
%       tau_guess - a first guess for the value of the decay constant of
%       the calcium indicator
%OUTPUT:
%       lambda - the estimated value of the parameter lambda
%       gama - the estimated value of the parameter gamma
%       est_spike - the estimated spike times at the chosen parameter
%             values

% setup for parallel computing
% myCluster = parcluster('local');
% 
% if getenv('ENVIRONMENT')
%    myCluster.JobStorageLocation = getenv('TMPDIR'); 
% end
% 
% parpool(myCluster,2);


if nargin<3
    tau_guess = 0.25;
end

possible_lambda = [0.01,0.1,0.25,0.5,0.75,1,1.25,1.5,2,3,4,5,6,7,8,10];

gama_guess = (1-1/(tau_guess*Fs))^2;

M = length(possible_lambda);
gammas = zeros(length(M),2);
MSEs = zeros(length(M),2);

ca_trace = ca_trace(:); % ensure that the calcium trace is a traditional (row) 
                        %  vector

N = length(ca_trace);
addcheck = 0;
% add a bin to make an even number N if needed
if mod(N,2) == 1
    ca_trace = [ca_trace;ca_trace(end)];
    N = N+1;
    addcheck = 1;
end

for fold=1:2
    if fold == 1
        trainInds = fold:2:N;
        testInds = fold+1:2:N;
    elseif fold == 2
        trainInds = fold:2:N;
        testInds = fold-1:2:N;
    end
    
    trainTrace = ca_trace(trainInds);
    testTrace = ca_trace(testInds);
    for m=1:M
        tic;
        [est_s] = L0_Algorithm(trainTrace,possible_lambda(m),gama_guess);
        test_time = toc;

        [best_gama] = FitGamma(est_s,trainTrace,gama_guess);
        
        [est_s] = L0_Algorithm(trainTrace,possible_lambda(m),best_gama);
        
        [estTrace] = GetEstTrace(trainTrace,est_s,best_gama);
        
       % estTrace = reshape([estTrace';zeros(size(estTrace'))],[],1);
       
       if fold == 1
           yy = zeros(length(testTrace),1);
           inds = [1,2];
           for ii=1:length(testTrace)-1
               yy(ii) = mean(estTrace(inds));
               inds = inds+1;
           end
           yy(end) = estTrace(end);
           
%            y = interp(estTrace,2);y = y(2:2:end);
       elseif fold == 2
           yy = zeros(length(testTrace),1);
           yy(1) = estTrace(1);
           inds = [1,2];
           for ii=2:length(testTrace)
              yy(ii) = mean(estTrace(inds));
              inds = inds+1;
           end
           
%            y = interp(estTrace,2);y = y(2:2:end);
%            y = [estTrace(1);y];
%            y = y(1:end-1);
       end
        
        estTrace = yy;
        gammas(m,fold) = best_gama;
        [MSEs(m,fold)] = GetMSE(estTrace,testTrace);

        % do not continue if the last run on the L0 algorithm took a long
        %  time, as increasing lambda values will tend to increase the time
        %  it takes to run, thus on the next iteration of the for loop, it
        %  will take even longer
        if test_time/(N/2) > 5e-3
            break;
        end
    end
end

%delete(gcp);

check = sum(MSEs==0,2);

indsToKeep = find(check==0);

gammas = gammas(indsToKeep,:);
MSEs = MSEs(indsToKeep,:);
possible_lambda = possible_lambda(indsToKeep);

gammas = mean(gammas,2);
MSEs = mean(MSEs,2);

[~,ind] = min(MSEs);
lambda = possible_lambda(ind);
gama = sqrt(gammas(ind));

if addcheck == 1
    ca_trace = ca_trace(1:end-1);
end

est_spike = L0_Algorithm(ca_trace,lambda,gama);
end

function [estTrace] = GetEstTrace(trueTrace,est_s,gama)
numFrames = length(trueTrace);
estTrace = zeros(numFrames,1);

for ii=2:numFrames
    if est_s(ii) > 0 
        ind = find(est_s(ii+1:end),1,'first');
        if isempty(ind) == 1
            nextSpike = numFrames+1;
        else
            nextSpike = ind+ii;
        end
        
        numer = 0;
        denom = 0;
        for jj=ii:nextSpike-1
            numer = numer+trueTrace(jj)*gama^(jj-ii);
            denom = denom+gama^(2*(jj-ii));
        end
        estTrace(ii) = numer/denom;
    else
        estTrace(ii) = gama*estTrace(ii-1);
    end
end

% another option to calculate the estimated trace (not exactly correct but
%  much faster) by convolving an exponential kernel with the estimated
%  spike train
% tauFs = 1/(1-gama);finalPoint = 0.005;
% maxTime = round(log(finalPoint)*(-tauFs));
% time = 0:maxTime;
% kernel = exp(-time/tauFs);
% estTrace = conv(est_s,kernel);
% estTrace = estTrace(1:numFrames);
end

function [gama] = FitGamma(est_s,trueTrace,gama_init)
   % fit values of gamma against the true trace with gradient descent
    numFrames = length(trueTrace);
    constantOnes = ones(numFrames,1);
    maxIter = 1e4;
    tolerance = 1e-6;
    currentGamma = gama_init;
    gamaStep = 1e-3;
    
    tauFs = 1/(1-currentGamma);finalPoint = 0.005;
    maxTime = round(log(finalPoint)*(-tauFs));
    time = (0:maxTime)';
    kernel = exp(-time/tauFs);
    estTrace = conv(est_s,kernel);estTrace = estTrace(1:numFrames);
    
    b = [constantOnes,estTrace]\trueTrace;
    currentMSE = var(b(1)*constantOnes+b(2)*estTrace-trueTrace);
    
    iter = 1;difference = 1;
    lineSteps = [gamaStep,1e-4,5e-3,1e-2,5e-2,1e-1];lineN = length(lineSteps)-1;
    tempMSE = zeros(lineN+1,1);
    tempGamma = zeros(lineN+1,1);
    while iter < maxIter && difference > tolerance
        tempGamma(1) = max(min(currentGamma+lineSteps(1),1-1e-3),0);
        
        tauFs = 1/(1-tempGamma(1));finalPoint = 0.005;
        maxTime = round(log(finalPoint)*(-tauFs));
        time = (0:maxTime)';
        kernel = exp(-time/tauFs);
        estTrace = conv(est_s,kernel);estTrace = estTrace(1:numFrames);
        
        b = [constantOnes,estTrace]\trueTrace;
        tempMSE(1) = var(b(1)*constantOnes+b(2)*estTrace-trueTrace);
        
        gradient = (tempMSE(1)-currentMSE)/(tempGamma(1)-currentGamma);
        
        % line search to see how far to move
        for ii=1:lineN
            tempGamma(ii+1) = max(min(currentGamma-sign(gradient)*lineSteps(ii+1),1-1e-3),0);
            tauFs = 1/(1-tempGamma(ii+1));finalPoint = 0.005;
            maxTime = round(log(finalPoint)*(-tauFs));
            time = (0:maxTime)';
            kernel = exp(-time/tauFs);
            estTrace = conv(est_s,kernel);estTrace = estTrace(1:numFrames);
            
            b = [constantOnes,estTrace]\trueTrace;
            tempMSE(ii+1) = var(b(1)*constantOnes+b(2)*estTrace-trueTrace);
        end
        [minMSE,ind] = min(tempMSE);
        if minMSE < currentMSE
            difference = currentMSE-minMSE;
            currentMSE = minMSE;
            currentGamma = tempGamma(ind);
        else
            difference = 1; 
        end
        iter = iter+1;
    end
    
    gama = currentGamma;
end

function [MSE] = GetMSE(estTrace,testTrace)
% use a simple model to calculate the deviance (which would be MSE as given
%  by the paper in the case of normally distributed data and the use of 
%  the estimated calcium trace as the only parameter in the model)
%  here we add an additional constant offset parameter, which should
%  actually do a better job if the original calcium trace is slightly
%  offset from zero mean
constantOnes = ones(length(estTrace),1);
b = [constantOnes,estTrace]\testTrace;
MSE = var(b(1)*constantOnes+b(2)*estTrace-testTrace);
end