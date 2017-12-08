function [] = NaturalImageTest()
% NaturalImageTest.m

cd ~/CloudStation/ByronExp/RestructuredNaturalImages/

load('NaturalImages.mat','allIms');
load('PinkResNaturalImages.mat','resIms');
load('UniformResNaturalImages.mat','unifResIms');

allIms = double(allIms);resIms = double(resIms);
unifResIms = double(unifResIms);
allIms = (allIms-mean(allIms(:)))/std(allIms(:));
resIms = (resIms-mean(resIms(:)))/std(resIms(:));
unifResIms = (unifResIms-mean(unifResIms(:)))/std(unifResIms(:));

N = 500;DIM = [100,100];

allIms = allIms(200:299,300:399,1:N);
resIms = resIms(200:299,300:399,1:N);
unifResIms = unifResIms(200:299,300:399,1:N);

[X,Y] = meshgrid(1:DIM(2),1:DIM(1));

filter = @(x) x(3).*exp(-(X-100).*(X-100)/(2*10*10)-(Y-100).*(Y-100)/(2*15*15))...
    .*sin(2*pi.*(cos(x(2)-pi/2).*X+sin(x(2)-pi/2).*Y).*x(1));

mask = zeros(DIM);
for ii=1:DIM(1)
    for jj=1:DIM(2)
        dist = sqrt((ii-100).^2+(jj-100).^2);
        if dist<30
            mask(ii,jj) = 1;
        end
    end
end
maskInds = find(mask);

a = 2;b = 2;c = 1;
sigmoid = @(x,a,b,c) a./(1+exp(-b.*(x-c)));

imrotations = [0,90,180,270];
spatialFrequencies = [1/100,1/75,1/50,1/40,1/30,1/20,1/10,1/5];
minLens = 1.*ones(length(spatialFrequencies),1);

resultNames = {'ASD_Nat','ASD_PoissNat','ASD_ResPinkNat','ASD_PoissResPinkNat',...
    'ASD_ResUnifNat','ASD_PoissResUnifNat','ASD_White','ASD_PoissWhite','ASD_Pink','ASD_PoissPink',...
    'STA_Nat','STA_PoissNat','STA_ResPinkNat','STA_PoissResPinkNat',...
    'STA_ResUnifNat','STA_PoissResUnifNat','STA_White',...
    'STA_PoissWhite','STA_Pink','STA_PoissPink'};

numConditions = length(resultNames);

biasOrient = zeros(length(spatialFrequencies),numConditions,2);
biasSpatFreq = zeros(length(spatialFrequencies),numConditions,2);
biasFull = zeros(length(spatialFrequencies),numConditions,2);

mseOrient = zeros(length(spatialFrequencies),numConditions,2);
mseSpatFreq = zeros(length(spatialFrequencies),numConditions,2);
mseFull = zeros(length(spatialFrequencies),numConditions,2);

numIter = 1000;

for jj=1:length(spatialFrequencies)
    spatFreq = spatialFrequencies(jj);
    minLen = minLens(jj);
    resultsOrient = zeros(numIter,numConditions);
    resultsSpatFreq = zeros(numIter,numConditions);
    resultsFull = zeros(numIter,numConditions);
    for kk=1:numIter
        
        % SIMULATE DATA
        orientation = rand*2*pi;
        currentFilter = filter([spatFreq,orientation,1]);
        filterNatOutput = zeros(N,1);
        filterResNatOutput = zeros(N,1);
        filterUnifResNatOutput = zeros(N,1);
        filterWhiteOutput = zeros(N,1);
        filterPinkOutput = zeros(N,1);
        
        naturalData = zeros(N,DIM(1)*DIM(2));
        resNatData = zeros(N,DIM(1)*DIM(2));
        unifResNatData = zeros(N,DIM(1)*DIM(2));
        whiteData = normrnd(0,1,[N,DIM(1)*DIM(2)]);
        pinkData = zeros(N,DIM(1)*DIM(2));
        for ii=1:N
            rotate = imrotations(randperm(4,1));
            temp1 = allIms(:,:,ii);
            temp2 = resIms(:,:,ii);
            temp3 = unifResIms(:,:,ii);
            
            temp1 = imrotate(temp1,rotate);
            temp2 = imrotate(temp2,rotate);
            temp3 = imrotate(temp3,rotate);
            
            temp4 = reshape(whiteData(ii,:),DIM);
            temp5 = spatialPattern(DIM+100,-2);
            temp5 = temp5(50:50+DIM(1)-1,50:50+DIM(2)-1);
            temp5 = (temp5-mean(temp5(:)))/std(temp5(:));
            
            naturalData(ii,:) = temp1(:);
            resNatData(ii,:) = temp2(:);
            unifResNatData(ii,:) = temp3(:);
            pinkData(ii,:) = temp5(:);
            
            filterNatOutput(ii) = sum(sum(temp1.*currentFilter));
            filterResNatOutput(ii) = sum(sum(temp2.*currentFilter));
            filterUnifResNatOutput(ii) = sum(sum(temp3.*currentFilter));
            filterWhiteOutput(ii) = sum(sum(temp4.*currentFilter));
            filterPinkOutput(ii) = sum(sum(temp5.*currentFilter));
        end
        
        filterNatOutput = filterNatOutput+normrnd(0,std(filterNatOutput)/5,[N,1]);
        filterResNatOutput = filterResNatOutput+normrnd(0,std(filterResNatOutput)/5,[N,1]);
        filterUnifResNatOutput = filterUnifResNatOutput+normrnd(0,std(filterUnifResNatOutput)/5,[N,1]);
        filterWhiteOutput = filterWhiteOutput+normrnd(0,std(filterWhiteOutput)/5,[N,1]);
        filterPinkOutput = filterPinkOutput+normrnd(0,std(filterPinkOutput)/5,[N,1]);
        
        filterNatOutput = filterNatOutput-mean(filterNatOutput);
        filterResNatOutput = filterResNatOutput-mean(filterResNatOutput);
        filterUnifResNatOutput = filterUnifResNatOutput-mean(filterUnifResNatOutput);
        filterWhiteOutput = filterWhiteOutput-mean(filterWhiteOutput);
        filterPinkOutput = filterPinkOutput-mean(filterPinkOutput);
        
        poissNat = filterNatOutput./std(filterNatOutput);
        poissResNat = filterResNatOutput./std(filterResNatOutput);
        poissUnifResNat = filterUnifResNatOutput./std(filterUnifResNatOutput);
        poissWhite = filterWhiteOutput./std(filterWhiteOutput);
        poissPink = filterPinkOutput./std(filterPinkOutput);
        
        poissNat = poissrnd(sigmoid(poissNat,a,b,c));
        poissNat = poissNat-mean(poissNat);
        poissResNat = poissrnd(sigmoid(poissResNat,a,b,c));
        poissResNat = poissResNat-mean(poissResNat);
        poissUnifResNat = poissrnd(sigmoid(poissUnifResNat,a,b,c));
        poissUnifResNat = poissUnifResNat-mean(poissUnifResNat);
        poissWhite = poissrnd(sigmoid(poissWhite,a,b,c));
        poissWhite = poissWhite-mean(poissWhite);
        poissPink = poissrnd(sigmoid(poissPink,a,b,c));
        poissPink = poissPink-mean(poissPink);
        
        
        % GET ESTIMATED FILTERS
        
        % natural images with asd
        [natFilt,~] = fastASD(naturalData,filterNatOutput,DIM,minLen);
        [poissNatFilt,~] = fastASD(naturalData,poissNat,DIM,minLen);
        
        % restructured pink natural images with asd
        [resNatFilt,~] = fastASD(resNatData,filterResNatOutput,DIM,minLen);
        [poissResNatFilt,~] = fastASD(resNatData,poissResNat,DIM,minLen);
        
        % restructured uniform natural images with asd
        [unifResNatFilt,~] = fastASD(unifResNatData,filterUnifResNatOutput,DIM,minLen);
        [poissUnifResNatFilt,~] = fastASD(unifResNatData,poissUnifResNat,DIM,minLen);
        
        % white noise with asd
        [whiteFilt,~] = fastASD(whiteData,filterWhiteOutput,DIM,minLen);
        [poissWhiteFilt,~] = fastASD(whiteData,poissWhite,DIM,minLen);
        
        % pink noise with asd
        [pinkFilt,~] = fastASD(pinkData,filterPinkOutput,DIM,minLen);
        [poissPinkFilt,~] = fastASD(pinkData,poissPink,DIM,minLen);
        
        % natural images with sta
        staNat = naturalData'*filterNatOutput;
        b = staNat\currentFilter(:);
        staNat = staNat*b;
        staPoissNat = naturalData'*poissNat;
        b = staPoissNat\currentFilter(:);
        staPoissNat = staPoissNat*b;
        
        % restructured pink natural images with sta
        staResNat = resNatData'*filterResNatOutput;
        b = staResNat\currentFilter(:);
        staResNat = staResNat*b;
        staPoissResNat = resNatData'*poissResNat;
        b = staPoissResNat\currentFilter(:);
        staPoissResNat = staPoissResNat*b;
        
        % restructured uniform natural images with sta
        staUnifResNat = unifResNatData'*filterUnifResNatOutput;
        b = staUnifResNat\currentFilter(:);
        staUnifResNat = staUnifResNat*b;
        staPoissUnifResNat = unifResNatData'*poissUnifResNat;
        b = staPoissUnifResNat\currentFilter(:);
        staPoissUnifResNat = staPoissUnifResNat*b;
        
        % white noise with sta
        staWhite = whiteData'*filterWhiteOutput;
        b = staWhite\currentFilter(:);
        staWhite = staWhite*b;
        staPoissWhite = whiteData'*poissWhite;
        b = staPoissWhite\currentFilter(:);
        staPoissWhite = staPoissWhite*b;
        
        % pink noise with sta
        staPink = pinkData'*filterPinkOutput;
        b = staPink\currentFilter(:);
        staPink = staPink*b;
        staPoissPink = pinkData'*poissPink;
        b = staPoissPink\currentFilter(:);
        staPoissPink = staPoissPink*b;
        
        
        % FIT GABOR FILTERS
        
        % natural images with asd
        temp = reshape(natFilt,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,1) = x(1)-spatFreq;
        resultsOrient(kk,1) = x(2)-orientation;
        resultsFull(kk,1) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        temp = reshape(poissNatFilt,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,2) = x(1)-spatFreq;
        resultsOrient(kk,2) = x(2)-orientation;
        resultsFull(kk,2) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        % restructured pink natural images with asd
        temp = reshape(resNatFilt,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,3) = x(1)-spatFreq;
        resultsOrient(kk,3) = x(2)-orientation;
        resultsFull(kk,3) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        temp = reshape(poissResNatFilt,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,4) = x(1)-spatFreq;
        resultsOrient(kk,4) = x(2)-orientation;
        resultsFull(kk,4) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        % restructured pink natural images with asd
        temp = reshape(unifResNatFilt,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,5) = x(1)-spatFreq;
        resultsOrient(kk,5) = x(2)-orientation;
        resultsFull(kk,5) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        temp = reshape(poissUnifResNatFilt,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,6) = x(1)-spatFreq;
        resultsOrient(kk,6) = x(2)-orientation;
        resultsFull(kk,6) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        % white noise with asd
        temp = reshape(whiteFilt,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,7) = x(1)-spatFreq;
        resultsOrient(kk,7) = x(2)-orientation;
        resultsFull(kk,7) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        temp = reshape(poissWhiteFilt,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,8) = x(1)-spatFreq;
        resultsOrient(kk,8) = x(2)-orientation;
        resultsFull(kk,8) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        % pink noise with asd
        temp = reshape(pinkFilt,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,9) = x(1)-spatFreq;
        resultsOrient(kk,9) = x(2)-orientation;
        resultsFull(kk,9) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        temp = reshape(poissPinkFilt,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,10) = x(1)-spatFreq;
        resultsOrient(kk,10) = x(2)-orientation;
        resultsFull(kk,10) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        % natural images with sta
        temp = reshape(staNat,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,11) = x(1)-spatFreq;
        resultsOrient(kk,11) = x(2)-orientation;
        resultsFull(kk,11) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        temp = reshape(staPoissNat,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,12) = x(1)-spatFreq;
        resultsOrient(kk,12) = x(2)-orientation;
        resultsFull(kk,12) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        % restructured pink natural images with sta
        temp = reshape(staResNat,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,13) = x(1)-spatFreq;
        resultsOrient(kk,13) = x(2)-orientation;
        resultsFull(kk,13) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        temp = reshape(staPoissResNat,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,14) = x(1)-spatFreq;
        resultsOrient(kk,14) = x(2)-orientation;
        resultsFull(kk,14) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        % restructured uniform natural images with sta
        temp = reshape(staUnifResNat,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,15) = x(1)-spatFreq;
        resultsOrient(kk,15) = x(2)-orientation;
        resultsFull(kk,15) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        temp = reshape(staPoissUnifResNat,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,16) = x(1)-spatFreq;
        resultsOrient(kk,16) = x(2)-orientation;
        resultsFull(kk,16) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        % white noise with sta
        temp = reshape(staWhite,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,17) = x(1)-spatFreq;
        resultsOrient(kk,17) = x(2)-orientation;
        resultsFull(kk,17) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        temp = reshape(staPoissWhite,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,18) = x(1)-spatFreq;
        resultsOrient(kk,18) = x(2)-orientation;
        resultsFull(kk,18) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        % pink noise with sta
        temp = reshape(staPink,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,19) = x(1)-spatFreq;
        resultsOrient(kk,19) = x(2)-orientation;
        resultsFull(kk,19) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        temp = reshape(staPoissPink,DIM);
        x = FitFilter(temp,[spatFreq,orientation,1],X,Y);
        resultsSpatFreq(kk,20) = x(1)-spatFreq;
        resultsOrient(kk,20) = x(2)-orientation;
        resultsFull(kk,20) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
    end
    
    for kk=1:numConditions
        biasSpatFreq(jj,kk,1) = mean(resultsSpatFreq(:,kk));
        biasOrient(jj,kk,1) = mean(resultsOrient(:,kk));
        biasFull(jj,kk,1) = mean(resultsFull(:,kk));
        biasSpatFreq(jj,kk,2) = std(resultsSpatFreq(:,kk));
        biasOrient(jj,kk,2) = std(resultsOrient(:,kk));
        biasFull(jj,kk,2) = std(resultsFull(:,kk));
        
        mseSpatFreq(jj,kk,1) = mean(resultsSpatFreq(:,kk).^2);
        mseOrient(jj,kk,1) = mean(resultsOrient(:,kk).^2);
        mseFull(jj,kk,1) = mean(resultsFull(:,kk).^2);
        mseSpatFreq(jj,kk,2) = std(resultsSpatFreq(:,kk).^2);
        mseOrient(jj,kk,2) = std(resultsOrient(:,kk).^2);
        mseFull(jj,kk,2) = std(resultsFull(:,kk).^2);
    end
    fprintf('Done with iteration %d\n',jj);
end

save Natural_Res_Ims-Test.mat mseSpatFreq mseOrient mseFull biasSpatFreq biasOrient ...
    resultNames spatialFrequencies DIM filter numConditions biasFull;
end

function [x] = FitFilter(y,x0,X,Y)
lb = [0,0,-Inf];ub = [Inf,Inf,Inf];
myFun = @(x) x(3).*exp(-(X-100).*(X-100)/(2*10*10)-(Y-100).*(Y-100)/(2*15*15))...
    .*sin(2*pi.*(cos(x(2)-pi/2).*X+sin(x(2)-pi/2).*Y).*x(1))-y;

x = lsqnonlin(myFun,x0,lb,ub);

end