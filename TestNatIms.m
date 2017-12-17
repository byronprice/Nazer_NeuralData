% TestNatIms.m

% simulation 3 from final paper for Nazer class
numIter = 1000;

load('NaturalImages.mat');

imMean = mean(double(allIms),3);

N = 750;DIM = [200,200];
dctDIM = 100*100;
fullData = zeros(N,prod(DIM));
dctData = zeros(N,dctDIM);
for ii=1:750
    temp = double(allIms(:,:,ii))-imMean;
    temp = temp(100:DIM(1)+100-1,100:DIM(2)+100-1);
    fullData(ii,:) = double(temp(:));

    R = mirt_dctn(temp);
    R = R(1:100,1:100);
    dctData(ii,:) = R(:);
end

% consider first getting rid of a bunch of values and then doing PCA
%  on data that has been Z-scored or on data that has had its 
%  mean divided out  

% S = cov(dctData); % or try shrinkage_cov
S = shrinkage_cov(dctData);
[V,D] = eig(S);

% choose dimensionality for dct images
allEigs = diag(D);
fullVariance = sum(allEigs);
for ii=10:10:dctDim
    start = dctDim-ii+1;
    eigenvals = allEigs(start:end);
    varianceProp = sum(eigenvals)/sum(fullVariance);
    if varianceProp <= 0.95
        break;
    end
end
q = length(allEigs(start:end));
meanEig = mean(allEigs(1:start-1));
W = V(:,start:end)*sqrtm(D(start:end,start:end)-meanEig.*eye(q));
W = fliplr(W);

dctData = dctData';
mu = mean(dctData,2);
dctData = dctData-repmat(mu,[1,N]);
x = pinv(W)*dctData;

reduceDctData = x'; % N-by-however-many-dimensions are kept
Sdct = cov(reduceDctData);

lPath = [10,5,4,3,2,1.5,1,0.75,0.5,0.25,0.1,0.05,0.02,0.01,0.001];
[XP,~,~,~,~,~] = QUIC('path',Sdct,50,lPath, 1e-8, 2, 100);

newXP = zeros(q,q,length(lPath)+1);
newXP(:,:,1:length(lPath)) = XP;
newXP(:,:,end) = (N-1).*pinv(reduceDctData'*reduceDctData);
XP = newXP;clear newXP;

result = struct('ASD',zeros(numIter,2),...
    'Sparse',zeros(numIter,2));

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
    spatFreq = rand*(2/30-2/75)+2/75;
    % SIMULATE DATA
    orientation = rand*2*pi;
    currentFilter = filter([spatFreq,orientation,1,normrnd(0,2),normrnd(0,2)]);
    null(jj) = norm(currentFilter);
    filterOutput = zeros(N,1);
    for kk=1:N
        temp = fullData(ii,:);temp = reshape(temp,DIM);
        filterOutput(kk) = sum(temp.*currentFilter);
    end
    filterOutput = filterOutput-mean(filterOutput);
    
    E = normrnd(0,std(filterOutput)/sqrt(2),[N,1]);
    filterOutputNoise = filterOutput+E;
    
    filterOutputNoise = filterOutputNoise-mean(filterOutputNoise);
    
    filterOutputSig = sigmoid(filterOutput,2,0,1);

    filterOutputSigmoid = poissrnd(filterOutputSigmoid.*gamrnd(3,1/3,[N,1]));
    meanSigmoid = mean(filterOutputSigmoid);
    filterOutputSigmoid = filterOutputSigmoid-meanSigmoid;
    
    
    [kestNoise,~,~] = fastASD(fullData,filterOutputNoise,DIM,2);
    [kestSig,~,~] = fastASD(fullData,filterOutputSigmoid,DIM,2);
    kestNoise = reshape(kestNoise,DIM);kestSig = reshape(kestSig,DIM);
    
    %result.ASD(jj,1) = (1/sqrt(prod(DIM)))*norm(kestNoise-currentFilter(:));
    result.ASD(jj,1) = (1/sqrt(maskSize))*norm(kestNoise(maskInds)-currentFilter(maskInds));
    
    %result.ASD(jj,2) = (1/sqrt(prod(DIM)))*norm(kestSig-currentFilter(:));
    result.ASD(jj,2) = (1/sqrt(maskSize))*norm(kestSig(maskInds)-currentFilter(maskInds));
    
    cv_result = zeros(length(lPath)+1,2);
    allInds = 1:N;cv_len = N-round(N/10);
    for zz=1:length(lPath)+1
        X = XP(:,:,zz);
        for yy=1:250
            inds = randperm(N,cv_len);
            inds = ismember(allInds,inds);
            holdOutInds = ~inds;
            suffStat = X*reduceDctData(inds,:)';
            
            tempestSparseNoise = suffStat*filterOutputNoise(inds);
            
            r = corrcoef(reduceDctData(holdOutInds,:)*tempestSparseNoise,filterOutputNoise(holdOutInds));
            
            cv_result(zz,1) = cv_result(zz,1)+r(1,2);
            
            tempestSparseSig = suffStat*filterOutputSigmoid(inds);
            
            r = corrcoef(max(reduceDctData(holdOutInds,:)*tempestSparseSig+meanSigmoid,0),filterOutputSigmoid(holdOutInds));
            
            cv_result(zz,2) = cv_result(zz,2)+r(1,2);
        end
    end
    
    
    [~,ind] = max(cv_result(:,1));
    X = XP(:,:,ind);
    suffStat = (1/(N-1)).*X*reduceDctData';
    estSparseNoise = suffStat*filterOutputNoise;

    estSparseNoise = W*estSparseNoise+mu;
    R = reshape(estSparseNoise,[100,100]);
    trueShape = zeros(DIM);
    trueShape(1:100,1:100) = R;

    estSparseNoise = mirt_idctn(trueShape);
    b = estSparseNoise(:)\currentFilter(:);estSparseNoise = b.*estSparseNoise;
    
    [~,ind] = max(cv_result(:,2));
    X = XP(:,:,ind);
    suffStat = (1./(N-1)).*X*reduceDctData';
    estSparseSig = suffStat*filterOutputSigmoid;

    estSparseSig = W*estSparseSig+mu;
    R = reshape(estSparseSig,[100,100]);
    trueShape = zeros(DIM);
    trueShape(1:100,1:100) = R;

    estSparseSig = mirt_idctn(trueShape);
    b = estSparseSig(:)\currentFilter(:);estSparseSig = b.*estSparseSig;
    
    %result.Sparse(jj,1) = (1/sqrt(prod(DIM)))*norm(estSparseNoise-currentFilter);
    result.Sparse(jj,1) = (1/sqrt(maskSize))*norm(estSparseNoise(maskInds)-currentFilter(maskInds));
    
    %result.Sparse(jj,2) = (1/sqrt(prod(DIM)))*norm(estSparseSig-currentFilter);
    result.Sparse(jj,2) = (1/sqrt(maskSize))*norm(estSparseSig(maskInds)-currentFilter(maskInds));
%         figure;plot(estSparse);hold on;plot(myfilter);
%         figure;plot(estSparseNoise);hold on;plot(myfilter);
%         figure;plot(estSparseSig);hold on;plot(myfilter);

end

save('NatImTests.mat','N','DIM','numIter','result');


data1 = result.ASD(:,1);
data2 = result.Sparse(:,1);
figure;histogram(data1);hold on;histogram(data2);
histogram(null);
legend('ASD','DCT','Null');title('Gaussian Encoding Model');
xlabel('RMSE');ylabel('Count');

data1 = result.ASD(:,2);
data2 = result.Sparse(:,2);
figure;histogram(data1);hold on;histogram(data2);
histogram(null);
legend('ASD','DCT','Null');title('Poisson-Gamma Encoding Model');
xlabel('RMSE');ylabel('Count');