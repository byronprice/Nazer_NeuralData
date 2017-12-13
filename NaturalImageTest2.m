function [] = NaturalImageTest2()
% NaturalImageTest2.m

cd ~/CloudStation/ByronExp/RestructuredNaturalImages/

load('NaturalImages.mat','allIms');
load('PinkResNaturalImages.mat','resIms');
load('UniformResNaturalImages.mat','unifResIms');

allIms = double(allIms);resIms = double(resIms);
unifResIms = double(unifResIms);
allIms = (allIms-mean(allIms(:)))/std(allIms(:));
resIms = (resIms-mean(resIms(:)))/std(resIms(:));
unifResIms = (unifResIms-mean(unifResIms(:)))/std(unifResIms(:));

N = 750;DIM = [100,100];

allIms = allIms(200:299,300:399,1:N);
resIms = resIms(200:299,300:399,1:N);
unifResIms = unifResIms(200:299,300:399,1:N);

newIms = zeros(N,413328);
level = 5;
for ii=1:N
    image = allIms(:,:,ii);
    [C,S] = wavedec2(image,level,'haar'); % 'db4' 
    newIms(ii,:) = C;
end

temp = sum(abs(newIms),1);
thresh = quantile(temp,0.5); % keep 25%
temp(temp<thresh) = 0;

indices = find(temp);

wdr_Ims = zeros(N,length(indices));

for ii=1:N
   wdr_Ims(ii,:) = newIms(ii,indices); 
end

S = cov(wdr_Ims);

[precision,~,~,~,~,~] = QUIC('default', S, 0.5, 1e-6, 2, 100);


[X,Y] = meshgrid(1:DIM(2),1:DIM(1));

filter = @(x) x(3).*exp(-(X-DIM(1)/2).^2/(2*10*10)-(Y-DIM(2)/2).^2/(2*15*15))...
    .*sin(2*pi.*(cos(x(2)-pi/2).*X+sin(x(2)-pi/2).*Y).*x(1));

mask = zeros(DIM);
for ii=1:DIM(1)
    for jj=1:DIM(2)
        dist = sqrt((ii-DIM(1)/2).^2+(jj-DIM(2)/2).^2);
        if dist<30
            mask(ii,jj) = 1;
        end
    end
end
maskInds = find(mask);maskSize = length(maskInds);

a = 2;bb = 2;c = 0;
sigmoid = @(x,a,b,c) a./(1+exp(-b.*(x-c)));

imrotations = [0,90,180,270];
N = [750,100,250,500,750];

numConditions = 6;
numIter = 250;
rmseFull = zeros(length(N),numConditions,numIter);

% myCluster = parcluster('local');
% 
% if getenv('ENVIRONMENT')
%    myCluster.JobStorageLocation = getenv('TMPDIR'); 
% end
% 
% parpool(myCluster,4);

for jj=1:length(N)
    resultsFull = zeros(numConditions,numIter);
    for kk=1:numIter
        spatFreq = rand*(1/30-1/75)+1/75;
        % SIMULATE DATA
        orientation = rand*2*pi;
        currentFilter = filter([spatFreq,orientation,1]);
        filterNatOutput = zeros(N(jj),1);
        filterResNatOutput = zeros(N(jj),1);
        filterUnifResNatOutput = zeros(N(jj),1);
%         filterWhiteOutput = zeros(N(jj),1);
%         filterPinkOutput = zeros(N(jj),1);
        
        naturalData = zeros(N(jj),DIM(1)*DIM(2));
        resNatData = zeros(N(jj),DIM(1)*DIM(2));
        unifResNatData = zeros(N(jj),DIM(1)*DIM(2));
%         whiteData = normrnd(0,1,[N(jj),DIM(1)*DIM(2)]);
%         pinkData = zeros(N(jj),DIM(1)*DIM(2));
        for ii=1:N(jj)
            rotate = imrotations(randperm(4,1));
            temp1 = allIms(:,:,ii);
            temp2 = resIms(:,:,ii);
            temp3 = unifResIms(:,:,ii);
            
            temp1 = imrotate(temp1,rotate);
            temp2 = imrotate(temp2,rotate);
            temp3 = imrotate(temp3,rotate);
            
%             temp4 = reshape(whiteData(ii,:),DIM);
%             temp5 = spatialPattern(DIM+100,-2);
%             temp5 = temp5(50:50+DIM(1)-1,50:50+DIM(2)-1);
%             temp5 = (temp5-mean(temp5(:)))/std(temp5(:));
            
            naturalData(ii,:) = temp1(:);
            resNatData(ii,:) = temp2(:);
            unifResNatData(ii,:) = temp3(:);
%             pinkData(ii,:) = temp5(:);
            
            filterNatOutput(ii) = sum(sum(temp1.*currentFilter));
            filterResNatOutput(ii) = sum(sum(temp2.*currentFilter));
            filterUnifResNatOutput(ii) = sum(sum(temp3.*currentFilter));
%             filterWhiteOutput(ii) = sum(sum(temp4.*currentFilter));
%             filterPinkOutput(ii) = sum(sum(temp5.*currentFilter));
        end
        
        filterNatOutput = filterNatOutput+normrnd(0,std(filterNatOutput)/sqrt(2),[N(jj),1]);
        filterResNatOutput = filterResNatOutput+normrnd(0,std(filterResNatOutput)/sqrt(2),[N(jj),1]);
        filterUnifResNatOutput = filterUnifResNatOutput+normrnd(0,std(filterUnifResNatOutput)/sqrt(2),[N(jj),1]);
%         filterWhiteOutput = filterWhiteOutput+normrnd(0,std(filterWhiteOutput)/sqrt(2),[N(jj),1]);
%         filterPinkOutput = filterPinkOutput+normrnd(0,std(filterPinkOutput)/sqrt(2),[N(jj),1]);
        
        filterNatOutput = filterNatOutput-mean(filterNatOutput);
        filterResNatOutput = filterResNatOutput-mean(filterResNatOutput);
        filterUnifResNatOutput = filterUnifResNatOutput-mean(filterUnifResNatOutput);
%         filterWhiteOutput = filterWhiteOutput-mean(filterWhiteOutput);
%         filterPinkOutput = filterPinkOutput-mean(filterPinkOutput);
        
        poissNat = filterNatOutput./std(filterNatOutput);
        poissResNat = filterResNatOutput./std(filterResNatOutput);
        poissUnifResNat = filterUnifResNatOutput./std(filterUnifResNatOutput);
%         poissWhite = filterWhiteOutput./std(filterWhiteOutput);
%         poissPink = filterPinkOutput./std(filterPinkOutput);
        
        poissNat = poissrnd(sigmoid(poissNat,a,bb,c).*gamrnd(2,0.5,[N(jj),1]));
        poissNat = poissNat-mean(poissNat);
        poissResNat = poissrnd(sigmoid(poissResNat,a,bb,c).*gamrnd(2,0.5,[N(jj),1]));
        poissResNat = poissResNat-mean(poissResNat);
        poissUnifResNat = poissrnd(sigmoid(poissUnifResNat,a,bb,c).*gamrnd(2,0.5,[N(jj),1]));
        poissUnifResNat = poissUnifResNat-mean(poissUnifResNat);
%         poissWhite = poissrnd(sigmoid(poissWhite,a,bb,c).*gamrnd(2,0.5,[N(jj),1]));
%         poissWhite = poissWhite-mean(poissWhite);
%         poissPink = poissrnd(sigmoid(poissPink,a,bb,c).*gamrnd(2,0.5,[N(jj),1]));
%         poissPink = poissPink-mean(poissPink);
%         
        
        % GET ESTIMATED FILTERS
        
        
        % FIT GABOR FILTERS
        
        % natural images with sta
        [kestNat,~,~] = fastASD(naturalData,filterNatOutput,DIM,4);
        temp = reshape(kestNat,DIM);
        resultsFull(1,kk) = (1/sqrt(maskSize))*norm(temp(maskInds)-currentFilter(maskInds));
        
        [kestPoissNat,~,~] = fastASD(naturalData,poissNat,DIM,4);
        temp = reshape(kestPoissNat,DIM);
        resultsFull(2,kk) = (1/sqrt(maskSize))*norm(temp(maskInds)-currentFilter(maskInds));
        
        % restructured pink natural images with sta
        [kestResNat,~,~] = fastASD(resNatData,filterResNatOutput,DIM,4);
        temp = reshape(kestResNat,DIM);
        resultsFull(3,kk) = (1/sqrt(maskSize))*norm(temp(maskInds)-currentFilter(maskInds));
        
        [kestPoissResNat,~,~] = fastASD(resNatData,poissResNat,DIM,4);
        temp = reshape(kestPoissResNat,DIM);
        resultsFull(4,kk) = (1/sqrt(maskSize))*norm(temp(maskInds)-currentFilter(maskInds));
        
        % restructured uniform natural images with sta
        [kestUnifResNat,~,~] = fastASD(unifResNatData,filterUnifResNatOutput,DIM,4);
        temp = reshape(kestUnifResNat,DIM);
        resultsFull(5,kk) = (1/sqrt(maskSize))*norm(temp(maskInds)-currentFilter(maskInds));
        
        [kestPoissUnifResNat,~,~] = fastASD(unifResNatData,poissUnifResNat,DIM,4);
        temp = reshape(kestPoissUnifResNat,DIM);
        resultsFull(6,kk) = (1/sqrt(maskSize))*norm(temp(maskInds)-currentFilter(maskInds));

    end
    
    rmseFull(jj,:,:) = resultsFull;
    fprintf('Done with iteration %d\n',jj);
end

save Natural_Res_Ims-ASDTest.mat rmseFull N DIM filter numConditions;
end

% function [x] = FitFilter(y,x0,X,Y)
% lb = [0,0,-Inf];ub = [Inf,Inf,Inf];
% myFun = @(x) x(3).*exp(-(X-100).*(X-100)/(2*10*10)-(Y-100).*(Y-100)/(2*15*15))...
%     .*sin(2*pi.*(cos(x(2)-pi/2).*X+sin(x(2)-pi/2).*Y).*x(1))-y;
% 
% x = lsqnonlin(myFun,x0,lb,ub);
% 
% end