% HW2.m
%  HW2 for Nazer's neural data class ... network inference from calcium
%   imaging data
% myCluster = parcluster('local');
% 
% if getenv('ENVIRONMENT')
%    myCluster.JobStorageLocation = getenv('TMPDIR'); 
% end
% 
% parpool(myCluster,2);

load('fluor_normal-4.mat');

reduceData = reduceData(5000:10000,:);
[N,numNeurons] = size(reduceData);


connectivityMatrix1 = zeros(numNeurons,numNeurons);
connectivityMatrix2 = zeros(numNeurons,numNeurons);
histParams = 10;
allInds = 1:numNeurons;
for ii=1:numNeurons
    Y = reduceData(21:end,ii);
    H = zeros(length(Y),histParams);
    for jj=1:histParams
       H(:,jj) = reduceData(histParams+1-jj:end-jj,ii); 
    end
    inds = find(allInds~=ii);
    Design = [ones(length(Y),1),H,reduceData(histParams+1:end,inds)];
    b = Design\Y;
    fullDev = sum((Design*b-Y).^2);
    
    tempConn1 = zeros(1,numNeurons);
    tempConn2 = zeros(1,numNeurons);
    for jj=1:numNeurons
        tempConn1(jj) = fullDev;
        
        newInds = find(allInds~=ii & allInds~=jj);
        Design = [ones(length(Y),1),H,reduceData(histParams+1:end,newInds)];
        b = Design\Y;dev = sum((Design*b-Y).^2);
        tempConn2(jj) = dev;
    end
    connectivityMatrix1(ii,:) = tempConn1;
    connectivityMatrix2(ii,:) = tempConn2;
end

save('Network4_Results.mat','connectivityMatrix1','connectivityMatrix2','numNeurons','N');

delete(gcp);

load('network_normal-4.mat');

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
fullParams = numNeurons+10;
restrictParams = numNeurons-1+10;
allInds = 1:numNeurons;
for ii=1:numNeurons
    inds = find(allInds~=ii);
    for jj=inds
        devFull = connectivityMatrix1(ii,jj);
        devRestrict = connectivityMatrix2(ii,jj);
        F = (devRestrict-devFull)/(devFull/(N-fullParams-1));
        PVALmat(ii,jj) = fcdf(F,1,N-fullParams,'upper');
    end
end

FDR = [0.001,0.01,0.05,0.1,0.25,0.5,0.75];
numTests = length(FDR);

truePositives = zeros(numTests,1);
falsePositives = zeros(numTests,1);
accuracy = zeros(numTests,1);

vectorPvals = sort(PVALmat(PVALmat~=0));
m = length(vectorPvals);
for ii=1:numTests
    % Benjamini-Hochberg
    alpha = FDR(ii);
    line = (1:m).*(alpha/m);
    difference = vectorPvals-line;
    index = find(difference<0,1,'last');
    pThreshold = vectorPvals(index);
    
    % threshold to get directed connectivity
    newPmat = PVALmat;newPmat(newPmat>pThreshold) = 0;
    newPmat(newPmat<=pThreshold) = 1;
    
    % convert to undirected connectivity
    undirPmat = newPmat;
    for jj=1:numNeurons
        for kk=1:numNeurons
             if undirPmat(jj,kk)==1
                 undirPmat(kk,jj) = 1;
             end
        end
    end
    
    % calculate error, etc.
    temp = undirPmat+undirConnectMat;
    accuracy(ii) = (sum(sum(temp==2))+sum(sum(temp==0)))/(numNeurons*numNeurons);
    
    % get true positive rate
    truePositives(ii) = sum(sum(temp==2))/sum(sum(undirConnectMat==1));
    % get false positive rate
    negatives = find(undirConnectMat==0);
    guessPositives = find(undirPmat==1);
    falsePositives(ii) = length(intersect(negatives,guessPositives))/length(negatives);
end

figure;
plot(falsePositives,truePositives,'LineWidth',2,'b');hold on;
plot(linspace(0,1,numTests),linspace(0,1,numTests),'--','k');
title(sprintf('ROC Curve'));
xlabel('False Positive Rate');ylabel('True Positive Rate');
legend('Granger F-test','Null');

[maxAc,ind] = max(accuracy);
fprintf('Maximum Accuracy at FDR %3.2f: %3.2f\n',FDR(ind),maxAc);