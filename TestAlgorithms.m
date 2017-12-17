% TestAlgorithms.m

% simulation 2 from final paper for Nazer class
numIter = 500;
N = [100,200,300,400,500,600,700,800,900,1000];DIM = 1000;

optimalPenalty = zeros(length(N),numIter,3);
result = struct('ASD',zeros(length(N),numIter,12),'Shrink',zeros(length(N),numIter,12),...
    'Sparse',zeros(length(N),numIter,12));

%  1,2 - white with zero-noise encoding, comma for full RMSE versus
%    only most important RMSE
%  3,4 - white with normal-noise encoding
%  5,6 - white with poisson-gamma encoding
%  7,8 - pink with zero-noise encoding
%  9,10 - pink with normal-noise encoding
%  11,12 - pink with poisson-gamma encoding
bb = 1e-5;t = 1:DIM;
importInds = 200:800;indLen = length(importInds);

sigmoid = @(x,a,b,c) a./(1+exp(-(x-b).*c));
for ii=1:length(N)
    for jj=1:numIter
        phi = 2*pi*rand-pi;
        f = ((20*rand)+5)/1000;
        myfilter = exp(-2*pi*bb.*(t-DIM/2).^2).*cos(2*pi*f.*(t-DIM/2)+phi);
        
        pinkData = zeros(N(ii),DIM);
        for kk=1:N(ii)
            temp = spatialPattern([2*DIM,1],-1);
            pinkData(kk,:) = temp(1:DIM);
        end
        pinkData = pinkData - repmat(mean(pinkData,1),[N(ii),1]);
        
        Spink = cov(pinkData);
        
        SpinkShrink = shrinkage_cov(pinkData);

        pinkShrink = pinv((N(ii)-1).*SpinkShrink);
        
        filterOutputPink = zeros(N(ii),1);
        for kk=1:N(ii)
            filterOutputPink(kk) = sum(pinkData(kk,:).*myfilter);
        end
        filterOutputPink = filterOutputPink-mean(filterOutputPink);
        
        E = normrnd(0,std(filterOutputPink)/sqrt(2),[N(ii),1]);
        filterOutputPinkNoise = filterOutputPink+E;
        
        filterOutputPinkNoise = filterOutputPinkNoise-mean(filterOutputPinkNoise);
        
        filterOutputPinkSigmoid = sigmoid(filterOutputPink,2,0,1);
        
        filterOutputPinkSigmoid = poissrnd(filterOutputPinkSigmoid.*gamrnd(3,1/3,[N(ii),1]));
        meanPinkSigmoid = mean(filterOutputPinkSigmoid);
        filterOutputPinkSigmoid = filterOutputPinkSigmoid-meanPinkSigmoid;
        
        
%         figure;plot(kest);hold on;plot(myfilter);
%         figure;plot(kestNoise);hold on;plot(myfilter);
%         figure;plot(kestSig);hold on;plot(myfilter);
        
        [kestP,~,~] = fastASD(pinkData,filterOutputPink,DIM,0);
        [kestNoiseP,~,~] = fastASD(pinkData,filterOutputPinkNoise,DIM,0);
        [kestSigP,~,~] = fastASD(pinkData,filterOutputPinkSigmoid,DIM,0);
        
        result.ASD(ii,jj,7) = (1/sqrt(DIM))*norm(kestP-myfilter');
        result.ASD(ii,jj,8) = (1/sqrt(indLen))*norm(kestP(importInds)-myfilter(importInds)');
        
        result.ASD(ii,jj,9) = (1/sqrt(DIM))*norm(kestNoiseP-myfilter');
        result.ASD(ii,jj,10) = (1/sqrt(indLen))*norm(kestNoiseP(importInds)-myfilter(importInds)');
        
        result.ASD(ii,jj,11) = (1/sqrt(DIM))*norm(kestSigP-myfilter');
        result.ASD(ii,jj,12) = (1/sqrt(indLen))*norm(kestSigP(importInds)-myfilter(importInds)');
        
        
%         figure;plot(estShrink);hold on;plot(myfilter);
%         figure;plot(estShrinkNoise);hold on;plot(myfilter);
%         figure;plot(estShrinkSig);hold on;plot(myfilter);
        
        suffStat = pinkShrink*pinkData';
        estShrink = suffStat*filterOutputPink;
        b = estShrink\myfilter';estShrink = estShrink.*b;
        
        estShrinkNoise = suffStat*filterOutputPinkNoise;
        b = estShrinkNoise\myfilter';estShrinkNoise = estShrinkNoise.*b;
        
        estShrinkSig = suffStat*filterOutputPinkSigmoid;
        b = estShrinkSig\myfilter';estShrinkSig = estShrinkSig.*b;
        
        result.Shrink(ii,jj,7) = (1/sqrt(DIM))*norm(estShrink-myfilter');
        result.Shrink(ii,jj,8) = (1/sqrt(indLen))*norm(estShrink(importInds)-myfilter(importInds)');
        
        result.Shrink(ii,jj,9) = (1/sqrt(DIM))*norm(estShrinkNoise-myfilter');
        result.Shrink(ii,jj,10) = (1/sqrt(indLen))*norm(estShrinkNoise(importInds)-myfilter(importInds)');
        
        result.Shrink(ii,jj,11) = (1/sqrt(DIM))*norm(estShrinkSig-myfilter');
        result.Shrink(ii,jj,12) = (1/sqrt(indLen))*norm(estShrinkSig(importInds)-myfilter(importInds)');
        
        
%         figure;plot(estSparse);hold on;plot(myfilter);
%         figure;plot(estSparseNoise);hold on;plot(myfilter);
%         figure;plot(estSparseSig);hold on;plot(myfilter);
        
        [XP,~,~,~,~,~] = QUIC('path', Spink,50,[2,1.5,1,0.75,0.5,0.2,0.1,1/25,1/50], 1e-6, 2, 100);
        
        temp = pinv(pinkData'*pinkData);
        
        newXP = zeros(DIM,DIM,10);
        newXP(:,:,1:9) = XP;newXP(:,:,end) = temp;
        
        cv_result = zeros(10,3);
        allInds = 1:N(ii);cv_len = N(ii)-round(N(ii)/10);
        for zz=1:10
            X = newXP(:,:,zz);
            for yy=1:100
                inds = randperm(N(ii),cv_len);
                inds = ismember(allInds,inds);
                holdOutInds = ~inds;
                suffStat = X*pinkData(inds,:)';
                
                tempestSparse = suffStat*filterOutputPink(inds);
                b = tempestSparse\myfilter';tempestSparse = tempestSparse.*b;
                
                r = corrcoef(pinkData(holdOutInds,:)*tempestSparse,filterOutputPink(holdOutInds));
                
                cv_result(zz,1) = cv_result(zz,1)+r(1,2);
                
                tempestSparseNoise = suffStat*filterOutputPinkNoise(inds);
                b = tempestSparseNoise\myfilter';tempestSparseNoise = tempestSparseNoise.*b;
                
                r = corrcoef(pinkData(holdOutInds,:)*tempestSparseNoise,filterOutputPinkNoise(holdOutInds));
                
                cv_result(zz,2) = cv_result(zz,2)+r(1,2);
                
                tempestSparseSig = suffStat*filterOutputPinkSigmoid(inds);
                b = tempestSparseSig\myfilter';tempestSparseSig = tempestSparseSig.*b;
                
                r = corrcoef(max(pinkData(holdOutInds,:)*tempestSparseSig+meanPinkSigmoid,0),filterOutputPinkSigmoid(holdOutInds));
                
                cv_result(zz,3) = cv_result(zz,3)+r(1,2);
            end
        end
        
        [~,ind] = max(cv_result(:,1));
        optimalPenalty(ii,jj,1) = ind;
        X = newXP(:,:,ind);
        suffStat = X*pinkData';
        estSparse = suffStat*filterOutputPink;
        b = estSparse\myfilter';estSparse = estSparse.*b;
        
        [~,ind] = max(cv_result(:,2));
        optimalPenalty(ii,jj,2) = ind;
        X = newXP(:,:,ind);
        suffStat = X*pinkData';
        estSparseNoise = suffStat*filterOutputPinkNoise;
        b = estSparseNoise\myfilter';estSparseNoise = estSparseNoise.*b;
        
        [~,ind] = max(cv_result(:,3));
        optimalPenalty(ii,jj,3) = ind;
        X = newXP(:,:,ind);
        suffStat = X*pinkData';
        estSparseSig = suffStat*filterOutputPinkSigmoid;
        b = estSparseSig\myfilter';estSparseSig = estSparseSig.*b;
        
        result.Sparse(ii,jj,7) = (1/sqrt(DIM))*norm(estSparse-myfilter');
        result.Sparse(ii,jj,8) = (1/sqrt(indLen))*norm(estSparse(importInds)-myfilter(importInds)');
        
        result.Sparse(ii,jj,9) = (1/sqrt(DIM))*norm(estSparseNoise-myfilter');
        result.Sparse(ii,jj,10) = (1/sqrt(indLen))*norm(estSparseNoise(importInds)-myfilter(importInds)');
        
        result.Sparse(ii,jj,11) = (1/sqrt(DIM))*norm(estSparseSig-myfilter');
        result.Sparse(ii,jj,12) = (1/sqrt(indLen))*norm(estSparseSig(importInds)-myfilter(importInds)');
%         figure;plot(estSparse);hold on;plot(myfilter);
%         figure;plot(estSparseNoise);hold on;plot(myfilter);
%         figure;plot(estSparseSig);hold on;plot(myfilter);

    end
    fprintf('\n\n\nDone with iter: %d\n\n\n',ii);
end

save('AlgorithmTests.mat','N','DIM','numIter','result','optimalPenalty');

for zz=[8,10,12]
    noNoise = zeros(3,10,3);
    gaussNoise = zeros(3,10,3);
    sigmoidNoise = zeros(3,10,3);
    
    
    for ii=1:10
        vals = squeeze(result.ASD(ii,:,zz));
        vals2 = squeeze(result.Shrink(ii,:,zz));
        vals3 = squeeze(result.Sparse(ii,:,zz));
        temp1 = zeros(numIter,1);temp2 = zeros(numIter,1);temp3 = zeros(numIter,1);
        for jj=1:numIter
            inds = random('Discrete Uniform',numIter,[numIter,1]);
            temp1(jj) = mean(vals(inds));
            temp2(jj) = mean(vals2(inds));
            temp3(jj) = mean(vals3(inds));
        end
        noNoise(1,ii,:) = quantile(temp1,[0.05/2,0.5,1-0.05/2]);
        noNoise(2,ii,:) = quantile(temp2,[0.05/2,0.5,1-0.05/2]);
        noNoise(3,ii,:) = quantile(temp3,[0.05/2,0.5,1-0.05/2]);
    end
    asd = squeeze(noNoise(1,:,:));
    shrink = squeeze(noNoise(2,:,:));
    sparse = squeeze(noNoise(3,:,:));
    
    figure;
    boundedline(N,asd(:,2),[asd(:,2)-asd(:,1),asd(:,3)-asd(:,2)],'-b','alpha');
    hold on;boundedline(N,shrink(:,2),[shrink(:,2)-shrink(:,1),shrink(:,3)-shrink(:,2)],'--g','alpha');
    boundedline(N,sparse(:,2),[sparse(:,2)-sparse(:,1),sparse(:,3)-sparse(:,2)],'om','alpha');
end