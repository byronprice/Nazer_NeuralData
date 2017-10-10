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

W = 5;
kernel = ones(W,1)./W;
correlations = zeros(length(train),1);

lambda = 1;gamma = 0.98;
for ii=train
    calcium_train = allData{ii,1};
    spike_train = allData{ii,2};
    
%     stimLen = length(calcium_train);
%     bin_edges = 0:5:stimLen;
   
   tempCorr = zeros(numNeurons(ii),1);
   for jj=1:numNeurons(ii)
      % calculate est_s here 
      [est_s] = L0_Algorithm(calcium_train(:,jj),lambda,gamma);
      est_s = conv(est_s,kernel);
      
      true_s = spike_train(:,jj);
      true_s = conv(true_s,kernel);
      
      [r,~] = corrcoef(true_s,est_s);
      tempCorr(jj) = r(1,2);
   end
   correlations(ii) = mean(tempCorr);
end

histogram(correlations)
mean(correlations)
std(correlations)
max(correlations)