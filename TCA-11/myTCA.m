function [Xs_new,Xt_new,A] = myTCA(X_src,X_tar,options)
%UNTITLED3 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    lambda = options.lambda;              
	dim = options.dim;                    
    X = [X_src',X_tar'];%
%     X = X * diag(sparse(1./sqrt(sum(X.^2))));
    X = X * diag(1./sqrt(sum(X.^2)));%��ÿһ������������һ��
    [m,n] = size(X);
    ns = size(X_src,1);
    nt = size(X_tar,1);
    e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
    L = e * e';
    H = eye(n)-1/n*ones(n,n);
    mat1 = X*L*X'+lambda*eye(m);
    mat2 = X*H*X';
    mat = pinv(mat1)*mat2;
    [A,~] = eigs(mat,dim);
    Z = A' * X;
%     Z = Z * diag(sparse(1./sqrt(sum(Z.^2))));
    Z = Z * diag(1./sqrt(sum(Z.^2)));%��ÿһ������������һ��
    Xs_new = Z(:,1:ns)';
    Xt_new = Z(:,ns+1:end)';
end

