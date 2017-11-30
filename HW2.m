% HW2.m
%  HW2 for Nazer's neural data class ... network inference from calcium
%   imaging data
myCluster = parcluster('local');

if getenv('ENVIRONMENT')
   myCluster.JobStorageLocation = getenv('TMPDIR'); 
end

parpool(myCluster,6);

load('small_net1-spikes.mat');

reduceData = spikeTrains(10001:60000,:);
[N,numNeurons] = size(reduceData);

connectivityMatrix1 = zeros(numNeurons,numNeurons);
connectivityMatrix2 = zeros(numNeurons,numNeurons);
histParams = 25;
allInds = 1:numNeurons;
parfor ii=1:numNeurons
    Y = reduceData(histParams+1:end,ii);
    H = zeros(length(Y),histParams);
    for jj=1:histParams
       H(:,jj) = reduceData(histParams+1-jj:end-jj,ii); 
    end

    Design = [ones(length(Y),1),H,reduceData(histParams+1:end,allInds~=ii)];
    [~,fullDev,~] = glmfit(Design,Y,'poisson','constant','off');
    
    tempConn1 = zeros(1,numNeurons);
    tempConn2 = zeros(1,numNeurons);
    for jj=1:numNeurons
        tempConn1(jj) = fullDev;
        
        newInds = find(allInds~=ii & allInds~=jj);
        Design = [ones(length(Y),1),H,reduceData(histParams+1:end,newInds)];
        [~,dev,~] = glmfit(Design,Y,'poisson','constant','off');
        tempConn2(jj) = dev;
    end
    connectivityMatrix1(ii,:) = tempConn1;
    connectivityMatrix2(ii,:) = tempConn2;
    fprintf('Done with neuron %d\n',ii);
end

save('SmallNet1_OASISResults.mat','connectivityMatrix1','connectivityMatrix2',...
    'numNeurons','N','histParams');

delete(gcp);

load('SmallNet1_OASISResults.mat');
load('network_small-net1.mat');

connectMat = zeros(numNeurons,numNeurons);

for ii=1:length(M)
   ind1 = M(ii,1);ind2 = M(ii,2);
   connectMat(ind1,ind2) = M(ii,3);
end

% (possibly) directed connectivity
connectMat(connectMat<0) = 0;

% guarantee we have undirected connectivity
undirConnectMat = connectMat;
for ii=1:numNeurons
    for jj=1:numNeurons
        if undirConnectMat(ii,jj) == 1
            undirConnectMat(jj,ii) = 1;
        end
    end
end

% F-test and FDR to determine undirected connectivity and get ROC curve

% get p-vals for each connection, then do Benjamini-Hochberg procedure with
%  p-vals in hand
PVALmat = zeros(numNeurons,numNeurons);
fullParams = numNeurons-1+histParams+1;
restrictParams = numNeurons-2+histParams+1;
paramDiff = fullParams-restrictParams;
allInds = 1:numNeurons;

devDiff = zeros(numNeurons,numNeurons);
for ii=1:numNeurons
    inds = find(allInds~=ii);
    for jj=inds
        devFull = connectivityMatrix1(ii,jj);
        devRestrict = connectivityMatrix2(ii,jj);
        chiSquare = devRestrict-devFull;
        temp = chi2cdf(chiSquare,paramDiff,'upper');
        devDiff(ii,jj) = chiSquare;
        if temp<1e-50
            temp = 1e-50;
        end
        PVALmat(ii,jj) = temp;
    end
end

FDR = [0,1e-6,1e-5,1e-4,0.001,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.999,1];
numTests = length(FDR);

truePositives = zeros(numTests,1);
falsePositives = zeros(numTests,1);
accuracy = zeros(numTests,1);

inds = ~diag(ones(numNeurons,1));
vectorPvals = sort(PVALmat(inds));
m = length(vectorPvals);
for ii=1:numTests
    % Benjamini-Hochberg
    alpha = FDR(ii);
    line = (1:m).*(alpha/m);
    difference = vectorPvals-line';
    index = find(difference<=0,1,'last');
    pThreshold = vectorPvals(index);
    
    if isempty(pThreshold) == 1
        pThreshold = min(vectorPvals)/2;
    end
    
    % threshold to get directed connectivity
    newPmat = PVALmat;newPmat(newPmat>=pThreshold) = 0;
    newPmat(newPmat>0) = 1;
    
    
    % convert to undirected connectivity
    undirPmat = newPmat;
    for jj=1:numNeurons
        for kk=1:numNeurons
             if undirPmat(jj,kk)==1
                 undirPmat(kk,jj) = 1;
             end
        end
    end
    onesMatrix = ones(size(undirPmat));
    inds = find(triu(onesMatrix,1));
    undirPvec = undirPmat(inds);
    undirConnVec = undirConnectMat(inds);
    
    % calculate error, etc.
    temp = undirPvec+undirConnVec;
    accuracy(ii) = (sum(temp==2)+sum(temp==0))/(numNeurons*numNeurons/2-numNeurons);
    
    % get true positive rate
    truePositives(ii) = sum(temp==2)/sum(undirConnVec==1);
    % get false positive rate
    negatives = find(undirConnVec==0);
    guessPositives = find(undirPvec==1);
    falsePositives(ii) = length(intersect(negatives,guessPositives))/length(negatives);
end

figure;
plot(falsePositives,truePositives,'b','LineWidth',2);hold on;
plot(linspace(0,1,numTests),linspace(0,1,numTests),'--k');
title(sprintf('ROC Curve: Granger Causality Raw Traces'));
xlabel('False Positive Rate');ylabel('True Positive Rate');
legend('Granger F-test','Null','Location','SouthEast');

AUC = trapz(falsePositives,truePositives);
fprintf('AUROCC: %3.2f\n',AUC);

[maxAc,ind] = max(accuracy);
fprintf('Maximum Accuracy at FDR %3.2e: %3.2f\n',FDR(ind),maxAc);

onesMatrix = ones(numNeurons,numNeurons);
inds = find(triu(onesMatrix,1));
temp = undirConnectMat(inds);

for ii=1:numNeurons
   for jj=1:numNeurons
       temp1 = devDiff(ii,jj);
       temp2 = devDiff(jj,ii);
       if temp2>temp1
           devDiff(ii,jj) = temp2;
       end
   end
end

devDiff = devDiff(inds);
devDiff1 = devDiff1(temp==1);
devDiff0 = devDiff0(temp==0);

figure;histogram(devDiff1,'normalization','probability');hold on;
histogram(devDiff0,'normalization','probability');
legend('Connection','No Connection','Location','Northwest');
title('Histogram of Chi-Square statistics');
xlabel('Chi-Square Statistic');
ylabel('Count');