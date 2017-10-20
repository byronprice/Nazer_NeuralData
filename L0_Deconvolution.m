% L0_Deconvolution.m

%  L0 normalization for the calcium fluorescence deconvolution problem from
%   Jewell & Witten 2017, Exact Spike-Train Inference Via L0 Optimization

% collect and organize the calcium imaging fluorescence traces
cd('~/Documents/Current-Projects/Nazer_NeuralData/spikefinder.train');

datasets = {'1','2','3','4','5','6','7','8','9','10'};
N = 10;
Fs = 100;

train = 1:5;test = 6:10;

allData = cell(N,2);
numNeurons = zeros(N,1);
for ii=1:10
   calcium_train = csvread([datasets{ii} '.train.calcium.csv']);
   spike_train = csvread([datasets{ii} '.train.spikes.csv']);  
   
   temp = quantile(calcium_train,[0.05,0.8]);

   calcium_train = (calcium_train-temp(1))./(temp(2)-temp(1));
   
   allData{ii,1} = calcium_train;
   allData{ii,2} = spike_train;
   numNeurons(ii) = size(calcium_train,2);
end

cd('~/Documents/Current-Projects/Nazer_NeuralData/');
W = 5;
kernel = ones(W,1)./W;
allCorrs = [];

for ii=1:10
    calcium_train = allData{ii,1};
    spike_train = allData{ii,2};
   
   tempCorr = zeros(numNeurons(ii),1);
   fprintf('Dataset %d\n',ii);
   for jj=1:numNeurons(ii)
      fprintf('   Neuron- %d\n',jj);
      % calculate est_s here
      [lambda,gama,est_s] = L0_CV(calcium_train(:,jj),Fs);

      est_s = conv(est_s,kernel);
      
      true_s = spike_train(:,jj);
      true_s = conv(true_s,kernel);
      
      [r,~] = corrcoef(true_s,est_s);
      tempCorr(jj) = r(1,2);
      allCorrs = [allCorrs;tempCorr(jj)];
      fprintf('   Corr: %3.3f\n',r(1,2));
      fprintf('   L: %3.2f G: %3.2f\n',lambda,gama);
   end
end

cd ~/CloudStation/ByronExp/
fileID = fopen('L0_Deconvolution-Results.txt','w');
fprintf(fileID,'Mean Corr: %3.2f\n',mean(allCorrs));
fprintf(fileID,'STD Corr: %3.2e\n',std(allCorrs));
fprintf(fileID,'Max Corr: %3.2f\n',max(allCorrs));
fprintf(fileID,'Min Corr: %3.2f\n\n',min(allCorrs));
fprintf(fileID,'%3.2f\n',allCorrs);

fclose(fileID);