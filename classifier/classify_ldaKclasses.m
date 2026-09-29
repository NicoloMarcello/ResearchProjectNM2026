function [cpred,Sw,L,K,muK,G] = classify_ldaKclasses(xtrain, ctrain, xtest,regularization)
% Multi-class classification using linear discriminant analysis 
% without prior
% INPUT:
%   xtrain : training set, p*ctr matrix with c datapoints in p dimensions
%   ctrain : 1*ctr vector with class labels corresponding to xtrain.
%   xtest  : test set, p*cte matrix with c datapoints in p dimensions 
%   OPTIONS: 
%       'regularization',0.03 
% OUTPUT:    
%   cpred  : 1*cte vector with predicted class labels for xtest.
%
% Joern Diedrichsen June 2010

if (nargin<4) 
    regularization=0.01;
end;

[P,N]=size(xtrain);     % size of training set 
classes=1:max(ctrain);  % classses we do classification on
cc=size(classes,2);     % class count

muK=zeros([P cc]);      % means
Sw=zeros(P,P);          % Within class variability 


%-------------calculate Parameter-----------------
for i=1:cc;
    j = find(ctrain==i);                                     % select datapoints in this class
    n = length(j);                                           % number of sampels per category 
    muK(:,i) = sum(xtrain(:,j),2)/n;                         % get the Cluster means 
    res = bsxfun(@minus,xtrain(:,j),muK(:,i));
    Sw = Sw+res*res';                         % Estimate common covariance matrix
end;
%-------------Regularisation----------------------
P=size(Sw,1);
Sw=Sw/N; 
Sw=Sw+eye(P)*trace(Sw)*regularization/P;

%-------------classify----------------------------
[dummy N_xtest] = size(xtest);
L=muK'/Sw;                                  % Calculate the classifier L(i,:)=muK'*inv(Sw)
K=sum(L.*muK',2);                           % constant term for each class muK'*inv(Sw)*muK
G=bsxfun(@plus,-0.5*K,L*xtest);             % Classification function for each class, test point 
[gmax,idx]=max(G); 
cpred=idx;