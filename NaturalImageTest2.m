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

N = 500;DIM = [200,200];

allIms = allIms(200:399,300:499,1:N);
resIms = resIms(200:399,300:499,1:N);
unifResIms = unifResIms(200:399,300:499,1:N);

powerSpectrum = zeros(DIM);

for ii=1:N
    temp = fft2(allIms(:,:,ii));
    powerSpectrum = powerSpectrum+abs(temp)./N;
end

powerSpectrum = 1./powerSpectrum;powerSpectrum(powerSpectrum==inf) = 0;
powerSpectrum = powerSpectrum.^0.5;

powerSpectrumUif = zeros(DIM);
for ii=1:N
    temp = fft2(unifResIms(:,:,ii));
    powerSpectrumUnif = powerSpectrumUnif+abs(temp)./N;
end

powerSpectrumUnif = 1./powerSpectrumUnif;powerSpectrumUnif(powerSpectrumUnif==inf) = 0;
powerSpectrumUnif = powerSpectrumUnif.^0.5;


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

a = 2;b = 2;c = 0;
sigmoid = @(x,a,b,c) a./(1+exp(-b.*(x-c)));

imrotations = [0,90,180,270];
spatialFrequencies = [1/100,1/75,1/50,1/40,1/30,1/20,1/10,1/5];

resultNames = {'STA_Nat','STA_PoissNat','STA_ResPinkNat','STA_PoissResPinkNat',...
    'STA_ResUnifNat','STA_PoissResUnifNat','STA_White',...
    'STA_PoissWhite','STA_Pink','STA_PoissPink','ub_STA_Nat','ub_STA_PoissNat','ub_STA_ResPinkNat',...
    'ub_STA_PoissResPinkNat','ub_STA_ResUnifNat','ub_STA_PoissResUnifNat',...
    'ub_STA_Pink','ub_STA_PoissPink'};

u = [(0:floor(DIM(1)/2)) -(ceil(DIM(1)/2)-1:-1:1)]'/DIM(1);
% Reproduce these frequencies along ever row
U = repmat(u,1,DIM(2)); 
% v is the set of frequencies along the second dimension.  For a square
% region it will be the transpose of u
v = [(0:floor(DIM(2)/2)) -(ceil(DIM(2)/2)-1:-1:1)]/DIM(2);
% Reproduce these frequencies along ever column
V = repmat(v,DIM(1),1);

% [U,V] = meshgrid(u,v); U = U'; V = V';

% Generate the power spectrum
S_f = (U.^2 + V.^2).^(BETA/2);

% Set any infinities to zero
S_f(S_f==inf) = 0;S_f = 1./S_f;S_f(S_f==inf) = 0;S_f = S_f.^0.5;

numConditions = length(resultNames);
numIter = 500;
rmseFull = zeros(length(spatialFrequencies),numConditions,numIter);

for jj=1:length(spatialFrequencies)
    spatFreq = spatialFrequencies(jj);
    resultsFull = zeros(numConditions,numIter);
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
        
        filterNatOutput = filterNatOutput+normrnd(0,std(filterNatOutput)/sqrt(2),[N,1]);
        filterResNatOutput = filterResNatOutput+normrnd(0,std(filterResNatOutput)/sqrt(2),[N,1]);
        filterUnifResNatOutput = filterUnifResNatOutput+normrnd(0,std(filterUnifResNatOutput)/sqrt(2),[N,1]);
        filterWhiteOutput = filterWhiteOutput+normrnd(0,std(filterWhiteOutput)/sqrt(2),[N,1]);
        filterPinkOutput = filterPinkOutput+normrnd(0,std(filterPinkOutput)/sqrt(2),[N,1]);
        
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
        
        poissNat = poissrnd(sigmoid(poissNat,a,b,c).*gamrnd(2,0.5,[N,1]));
        poissNat = poissNat-mean(poissNat);
        poissResNat = poissrnd(sigmoid(poissResNat,a,b,c).*gamrnd(2,0.5,[N,1]));
        poissResNat = poissResNat-mean(poissResNat);
        poissUnifResNat = poissrnd(sigmoid(poissUnifResNat,a,b,c).*gamrnd(2,0.5,[N,1]));
        poissUnifResNat = poissUnifResNat-mean(poissUnifResNat);
        poissWhite = poissrnd(sigmoid(poissWhite,a,b,c).*gamrnd(2,0.5,[N,1]));
        poissWhite = poissWhite-mean(poissWhite);
        poissPink = poissrnd(sigmoid(poissPink,a,b,c).*gamrnd(2,0.5,[N,1]));
        poissPink = poissPink-mean(poissPink);
        
        
        % GET ESTIMATED FILTERS
        
        % natural images with sta
        staNat = naturalData'*filterNatOutput;
        
        ubstaNat = reshape(staNat,DIM);
        ubstaNat = real(ifft2(fft2(ubstaNat).*powerSpectrum));
        b = ubstaNat(:)\currentFilter(:);
        ubstaNat = ubstaNat*b;
        
        b = staNat\currentFilter(:);
        staNat = staNat*b;
        
        
        
        staPoissNat = naturalData'*poissNat;
        
        ubstaPoissNat = reshape(staPoissNat,DIM);
        ubstaPoissNat = real(ifft2(fft2(ubstaPoissNat).*powerSpectrum));
        b = ubstaPoissNat(:)\currentFilter(:);
        ubstaPoissNat = ubstaNat*b;
        
        b = staPoissNat\currentFilter(:);
        staPoissNat = staPoissNat*b;
        
        % restructured pink natural images with sta
        staResNat = resNatData'*filterResNatOutput;
        
        ubstaResNat = reshape(staResNat,DIM);
        ubstaNat = real(ifft2(fft2(ubstaResNat).*S_f));
        b = ubstaResNat(:)\currentFilter(:);
        ubstaResNat = ubstaResNat*b;
        
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
        
        % natural images with sta
        temp = reshape(staNat,DIM);
        resultsFull(kk,11) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        temp = reshape(staPoissNat,DIM);
        resultsFull(kk,12) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        % restructured pink natural images with sta
        temp = reshape(staResNat,DIM);
        resultsFull(kk,13) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        temp = reshape(staPoissResNat,DIM);
        resultsFull(kk,14) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        % restructured uniform natural images with sta
        temp = reshape(staUnifResNat,DIM);
        resultsFull(kk,15) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        temp = reshape(staPoissUnifResNat,DIM);
        resultsFull(kk,16) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        % white noise with sta
        temp = reshape(staWhite,DIM);
        resultsFull(kk,17) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        temp = reshape(staPoissWhite,DIM);
        resultsFull(kk,18) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        % pink noise with sta
        temp = reshape(staPink,DIM);
        resultsFull(kk,19) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
        
        temp = reshape(staPoissPink,DIM);
        resultsFull(kk,20) = mean(abs(temp(maskInds)-currentFilter(maskInds)));
    end
    
    rmseFull(jj,:,:) = resultsFull;
    fprintf('Done with iteration %d\n',jj);
end

save Natural_Res_Ims-Test.mat rmseFull ...
    resultNames spatialFrequencies DIM filter numConditions;
end

function [x] = FitFilter(y,x0,X,Y)
lb = [0,0,-Inf];ub = [Inf,Inf,Inf];
myFun = @(x) x(3).*exp(-(X-100).*(X-100)/(2*10*10)-(Y-100).*(Y-100)/(2*15*15))...
    .*sin(2*pi.*(cos(x(2)-pi/2).*X+sin(x(2)-pi/2).*Y).*x(1))-y;

x = lsqnonlin(myFun,x0,lb,ub);

end