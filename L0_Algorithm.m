function [est_s] = L0_Algorithm(ca_trace,lambda,gama)
%L0_Algorithm.m
%  Runs the algorithm from Jewell & Witten 2017, calculating the spike times
%   for a given calcium fluorescence trace

% the free parameter gama depends on the acquisition frame rate and does
%  not have any physical interpretation ... if you prefer a parameter that
%  does have a physical meaning, then you can take tau 
%  (the decay constant of the calcium indicator) as
%           gamma = 1 - 1/(tau*Fs)
%   or      tau = 
%  where Fs is the sampling frequency of the imaging system

N = length(ca_trace);
F = zeros(N+1,1);
est_s = zeros(N,1);
 
changePoints = zeros(N,1);

F(1) = -lambda;

[D_ab,C_ab] = CalcD1(ca_trace(1),gama);
F(2) = F(1)+(D_ab(1)-(C_ab(1)/C_ab(2))*D_ab(2)+((C_ab(1)/C_ab(2))^2/2)*D_ab(3))+lambda;

changePoints(1) = 0;

epsilon = [1;2];
Darray = [D_ab,C_ab,1,1]; % hold on to these for future iterations
for ii=3:N+1
    %fprintf('%d\n',ii);
    
    set = zeros(length(epsilon),1);
    tempArray = zeros(length(epsilon),7);
    count = 1;
    for jj=epsilon'
        a = epsilon(count);b = ii-1;
        check = sum(ismember(Darray(:,6:7),[a,b-1]),2);
        inds = find(check==2,1,'first');
        if isempty(inds) == 1
            [D_ab,C_ab] = CalcD1(ca_trace(jj:ii-1),gama);
        else
            [D_ab,C_ab] = CalcDplus1(Darray(inds,:),ca_trace(ii-1),gama);
        end
        set(count) = F(jj)+(D_ab(1)-(C_ab(1)/C_ab(2))*D_ab(2)+((C_ab(1)/C_ab(2))^2/2)*D_ab(3))+lambda;
        tempArray(count,:) = [D_ab,C_ab,a,b];
        count = count+1;
    end
    [F(ii),ind] = min(set);
    changePoints(ii-1) = epsilon(ind);
    
    toKeep = (set-lambda)<F(ii);
    epsilon = [epsilon(toKeep);ii];
    Darray = tempArray(toKeep,:);
end
changePoints = changePoints(changePoints~=0);
est_s(changePoints) = 1;
ind = find(est_s==1,1,'first');est_s(ind) = 0;

end

function [D_ab,C_ab] = CalcD1(y,gama)
% calculate D{y(a:b)} as in equation 8 from the paper
N = length(y);

C_ab = zeros(1,2);
C_ab(1) = y(1);
C_ab(2) = 1;
for ii=2:N
    C_ab(1) = C_ab(1)+y(ii)*gama^(ii-1);
    C_ab(2) = C_ab(2)+gama^(2*(ii-1));
end

D_ab = zeros(1,3);
for ii=1:N
    D_ab(1) = D_ab(1)+y(ii)^2/2;
    D_ab(2) = D_ab(2)+y(ii)*gama^(ii-1);
    D_ab(3) = D_ab(3)+gama^(2*(ii-1));
end

end

function [D_ab,C_ab] = CalcDplus1(Darray,y,gama)
% calculate D{y(a:b+1)} as in equation 8 from the paper

D_ab = Darray(1:3);
C_ab = Darray(4:5);

D_ab(1) = D_ab(1)+y^2/2;
D_ab(2) = D_ab(2)+y*gama^(Darray(end)-Darray(end-1));
D_ab(3) = D_ab(3)+gama^(2*(Darray(end)-Darray(end-1)));

C_ab(1) = C_ab(1)+y*gama^(Darray(end)-Darray(end-1));
C_ab(2) = C_ab(2)+gama^(2*(Darray(end)-Darray(end-1)));


end

