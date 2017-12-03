function [lambda,b,deviance] = L0Regression(Design,Y,inds,lambdaVec,N,b0)
% L0Regression.m
%  for this one, a good starting point is probably to knock out a bunch of
%   the maximum likelihood values first
train = round(N*0.7);

options = optimoptions('fmincon','MaxFunctionEvaluations',8000);
nLambda = length(lambdaVec);
fullError = zeros(nLambda,1);
bVals = zeros(length(b0),nLambda);
count = 1;
for lambda = lambdaVec
    myFun = @(x) Lasso(x,lambda,Design(1:train,:),Y(1:train),inds);
    b = fmincon(myFun,b0,[],[],[],[],[],[],[],options);
    fullError(count) = sum((Design(train+1:end,:)*b-Y(train+1:end)).^2);
    bVals(:,count) = b;
    count = count+1;
end

[deviance,ind] = min(fullError);
b = bVals(:,ind);
lambda = lambdaVec(ind);

% myFun = @(x) Lasso(x,lambda,Design,Y,inds);
% b = fmincon(myFun,b0,[],[],[],[],[],[],[],options);
% 
% deviance = sum((Design*b-Y).^2);

end

function [error] = Lasso(b,lambda,Design,Y,inds)
   
error = sum((Design*b-Y).^2)+lambda.*sum(b(inds)~=0);

end