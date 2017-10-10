function [est_s] = L0_Algorithm(ca_trace,lambda,gama)
%L0_Algorithm.m
%  Runs the algorithm from Jewell & Witten 2017, calculating the spike times
%   for a given calcium fluorescence trace

N = length(ca_trace);
F = zeros(N+1,1);
est_s = zeros(N,1);

% 
changePoints = zeros(N,1);

% 
F(1) = -lambda;

F(2) = F(1)+CalcD(ca_trace(1),gama)+lambda;
changePoints(1) = 0;
epsilon = [1;2];
for ii=3:N+1
    %fprintf('%d\n',ii);
    
    set = zeros(length(epsilon),1);
    count = 1;
    for jj=epsilon'
        set(count) = F(jj)+CalcD(ca_trace(jj:ii-1),gama)+lambda;
        count = count+1;
    end
    [F(ii),ind] = min(set);
    changePoints(ii-1) = epsilon(ind);
    
    epsilon = [epsilon((set-lambda)<F(ii));ii];
end
changePoints = changePoints(changePoints~=0);
est_s(changePoints) = 1;
ind = find(est_s==1,1,'first');est_s(ind) = 0;

end

function [D_ab] = CalcD(y,gama)
% calculate D{y(a:b)} as in equation 8 from the paper
N = length(y);

C_ab_num = y(1);
C_ab_denom = 1;
for ii=2:N
    C_ab_num = C_ab_num+y(ii)*gama^(ii-1);
    C_ab_denom = C_ab_denom+gama^(2*(ii-1));
end

C_ab = C_ab_num/C_ab_denom;
D_ab1 = 0;D_ab2 = 0;D_ab3 = 0;
for ii=1:N
    D_ab1 = D_ab1+y(ii)^2/2;
    D_ab2 = D_ab2+y(ii)*gama^(ii-1);
    D_ab3 = D_ab3+gama^(2*(ii-1));
end

D_ab = D_ab1-C_ab*D_ab2+(C_ab^2/2)*D_ab3;

end

