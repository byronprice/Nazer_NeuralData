% TestNatIms.m

% simulation 3 from final paper for Nazer class
numIter = 1000;

load('NaturalImages.mat');

imMean = mean(double(allIms),3);

N = 750;DIM = [200,200];
dctDIM = 125*125;
fullData = zeros(N,prod(DIM));
dctData = zeros(N,dctDIM);
for ii=1:750
    temp = double(allIms(:,:,ii))-imMean;
    temp = temp(100:DIM(1)+100-1,100:DIM(2)+100-1);
    fullData(ii,:) = double(temp(:));

    R = mirt_dctn(temp);
    R = R(1:125,1:125);
    dctData(ii,:) = R(:);
end

clear allIms;

% consider first getting rid of a bunch of values and then doing PCA
%  on data that has been Z-scored or on data that has had its 
%  mean divided out  

S = cov(dctData); % or try shrinkage_cov
%S = shrinkage_cov(dctData);
[V,D] = eig(S);

pcaData = dctData';
mu = mean(pcaData,2);
pcaData = pcaData-repmat(mu,[1,N]);


% allEigs = diag(D);
% fullVariance = sum(allEigs);
% for zz=5:1:dctDIM
%     start = dctDIM-zz+1;
%     eigenvals = allEigs(start:end);
%     varianceProp = sum(eigenvals)/fullVariance;
%     if varianceProp >= 0.999
%         break;
%     end
% end
% q = length(allEigs(start:end));
% meanEig = mean(allEigs(1:start-1));
% W = V(:,start:end)*sqrtm(D(start:end,start:end)-meanEig.*eye(q));
% W = fliplr(W);
% 
% x = pinv(W)*pcaData;
% sparseDctData = x'; % N-by-however-many-dimensions are kept
% 
% sparseCov = cov(sparseDctData);
% lPath

result = struct('ASD',zeros(numIter,2),...
    'Sparse',zeros(numIter,2),'PCA',zeros(numIter,2));

[X,Y] = meshgrid(1:DIM(2),1:DIM(1));

filter = @(x) x(3).*exp(-(X-DIM(1)/2+x(4)).^2/(2*15*15)-(Y-DIM(2)/2+x(5)).^2/(2*15*15))...
    .*sin(2*pi.*(cos(x(2)-pi/2).*X+sin(x(2)-pi/2).*Y).*x(1));

a = 3;bb = 2;c = 1;
sigmoid = @(x,a,b,c) a./(1+exp(-b.*(x-c)));

mask = zeros(DIM);
for ii=1:DIM(1)
    for jj=1:DIM(2)
        dist = sqrt((ii-DIM(1)/2).^2+(jj-DIM(2)/2).^2);
        if dist<40
            mask(ii,jj) = 1;
        end
    end
end
maskInds = find(mask);maskSize = length(maskInds);


null = zeros(numIter,1);
for jj=1:numIter
    spatFreq = rand*(4/75-1/75)+1/75;
    % SIMULATE DATA
    orientation = rand*2*pi;
    currentFilter = filter([spatFreq,orientation,1,normrnd(0,2),normrnd(0,2)]);
    null(jj) = norm(currentFilter);
    filterOutput = zeros(N,1);asdDIM = 100;
    forASD = zeros(N,asdDIM*asdDIM);
    for kk=1:N
        temp = fullData(kk,:);temp = reshape(temp,DIM);
        smallTemp = imresize(temp,asdDIM(1)/DIM(1));
        forASD(kk,:) = smallTemp(:);
        filterOutput(kk) = sum(sum(temp.*currentFilter));
    end
    filterOutput = filterOutput-mean(filterOutput);
    
    E = normrnd(0,std(filterOutput)/sqrt(2),[N,1]);
    filterOutputNoise = filterOutput+E;
    
    filterOutputNoise = filterOutputNoise-mean(filterOutputNoise);
    
    filterOutputSigmoid = sigmoid(filterOutput,2,0,1);

    filterOutputSigmoid = poissrnd(filterOutputSigmoid.*gamrnd(3,1/3,[N,1]));
    meanSigmoid = mean(filterOutputSigmoid);
    filterOutputSigmoid = filterOutputSigmoid-meanSigmoid;
    
    
    [kestNoise,~,~] = fastASD(forASD,filterOutputNoise,[asdDIM,asdDIM],2);
    [kestSig,~,~] = fastASD(forASD,filterOutputSigmoid,[asdDIM,asdDIM],2);
    
    r1 = corrcoef(forASD*kestNoise,filterOutput);
    r2 = corrcoef(forASD*kestSig,filterOutput);
    
    %result.ASD(jj,1) = (1/sqrt(prod(DIM)))*norm(kestNoise-currentFilter(:));
    result.ASD(jj,1) = r1(1,2);
    
    %result.ASD(jj,2) = (1/sqrt(prod(DIM)))*norm(kestSig-currentFilter(:));
    result.ASD(jj,2) = r2(1,2);
    
    
    %result.Sparse(jj,1) = (1/sqrt(prod(DIM)))*norm(estSparseNoise-currentFilter);
%     result.Sparse(jj,1) = (1/sqrt(maskSize))*norm(estSparseNoise(maskInds)-currentFilter(maskInds));
%     
%     %result.Sparse(jj,2) = (1/sqrt(prod(DIM)))*norm(estSparseSig-currentFilter);
%     result.Sparse(jj,2) = (1/sqrt(maskSize))*norm(estSparseSig(maskInds)-currentFilter(maskInds));
    
    % choose dimensionality for dct images
    allEigs = diag(D);
    fullVariance = sum(allEigs);
    saveVar = 0.7:0.01:0.99;
    
    medianCorrs = zeros(length(saveVar),2);
    count = 1;
    for yy=saveVar
        for zz=5:1:dctDIM
            start = dctDIM-zz+1;
            eigenvals = allEigs(start:end);
            varianceProp = sum(eigenvals)/fullVariance;
            if varianceProp >= yy
                break;
            end
        end
        q = length(allEigs(start:end));
        meanEig = mean(allEigs(1:start-1));
        W = V(:,start:end)*sqrtm(D(start:end,start:end)-meanEig.*eye(q));
        W = fliplr(W);
     
        x = pinv(W)*pcaData;
        
        reduceDctData = x'; % N-by-however-many-dimensions are kept
        
        allInds = 1:N;cv_iter = 250;
        tempCorrs = zeros(cv_iter,2);cv_len = round(N*0.9);
        for zz=1:cv_iter
            inds = randperm(N,cv_len);
            inds = ismember(allInds,inds);
            holdOutInds = ~inds;
            
            estNoise = reduceDctData(inds,:)\filterOutputNoise(inds);
            r = corrcoef(reduceDctData(holdOutInds,:)*estNoise,filterOutputNoise(holdOutInds));
            
            tempCorrs(zz,1) = r(1,2);
            
            estSig = reduceDctData(inds,:)\filterOutputSigmoid(inds);
            r = corrcoef(reduceDctData(holdOutInds,:)*estSig,filterOutputSigmoid(holdOutInds));
            
            tempCorrs(zz,2) = r(1,2);
        end
        medianCorrs(count,1) = median(tempCorrs(:,1));
        medianCorrs(count,2) = median(tempCorrs(:,2));
        count = count+1;
    end
    [~,ind1] = max(medianCorrs(:,1));
    [~,ind2] = max(medianCorrs(:,2));
    
    for zz=5:1:dctDIM
        start = dctDIM-zz+1;
        eigenvals = allEigs(start:end);
        varianceProp = sum(eigenvals)/fullVariance;
        if varianceProp >= saveVar(min(ind1+1,length(saveVar)))
            break;
        end
    end
    q = length(allEigs(start:end));
    meanEig = mean(allEigs(1:start-1));
    W = V(:,start:end)*sqrtm(D(start:end,start:end)-meanEig.*eye(q));
    W = fliplr(W);
    
    x = pinv(W)*pcaData;
    
    reduceDctData = x'; % N-by-however-many-dimensions are kept
    estNoise = reduceDctData\filterOutputNoise;
    
    r1 = corrcoef(reduceDctData*estNoise,filterOutput);
    
    for zz=5:1:dctDIM
        start = dctDIM-zz+1;
        eigenvals = allEigs(start:end);
        varianceProp = sum(eigenvals)/fullVariance;
        if varianceProp >= saveVar(min(ind2+1,length(saveVar)))
            break;
        end
    end
    q = length(allEigs(start:end));
    meanEig = mean(allEigs(1:start-1));
    W = V(:,start:end)*sqrtm(D(start:end,start:end)-meanEig.*eye(q));
    W = fliplr(W);
    
    x = pinv(W)*pcaData;
    
    reduceDctData = x'; % N-by-however-many-dimensions are kept
    
    estSig = reduceDctData\filterOutputSigmoid;
    
    r2 = corrcoef(reduceDctData*estSig,filterOutput);
    
    %result.PCA(jj,1) = (1/sqrt(prod(DIM)))*norm(estSparseNoise-currentFilter);
    result.PCA(jj,1) = r1(1,2);
    
    %result.PCA(jj,2) = (1/sqrt(prod(DIM)))*norm(estSparseSig-currentFilter);
    result.PCA(jj,2) = r2(1,2);
end

save('NatImTests.mat','N','DIM','numIter','result');


data1 = result.ASD(:,1);
data2 = result.Sparse(:,1);
figure;histogram(data1);hold on;histogram(data2);
histogram(null);
legend('ASD','DCT','Null');title('Gaussian Encoding Model');
xlabel('Correlation Coefficient');ylabel('Count');

data1 = result.ASD(:,2);
data2 = result.Sparse(:,2);
figure;histogram(data1);hold on;histogram(data2);
histogram(null);
legend('ASD','DCT','Null');title('Poisson-Gamma Encoding Model');
xlabel('Correlation Coefficient');ylabel('Count');