function [Xs_new,Xt_new,A] = myTCA(X_src,X_tar,options)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
%%  参数设置
    lambda = options.lambda;              
	dim = options.dim;                    
	kernel_type = options.kernel_type;    
	gamma = options.gamma; 
%%  计算过程
    X = [X_src',X_tar'];%
    X = X * diag(sparse(1./sqrt(sum(X.^2))));
%     X = X * diag(1./sqrt(sum(X.^2)));%对每一列做二范数归一化
    [m,n] = size(X);
    ns = size(X_src,1);
    nt = size(X_tar,1);
    e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
    L = e * e';
    L = L / norm(L,'fro');%这一部会影响精度。
    H = eye(n)-1/n*ones(n,n);
    
    if strcmp(kernel_type,'primal')
        mat1 = X*L*X'+lambda*eye(m);
        mat2 = X*H*X';
        [A,~] = eigs(pinv(mat1)*mat2,dim);
        %[A,~] = eigs(mat1,mat2,dim,'SM');与上面等价
        Z = A' * X;
    else
        K = TCA_kernel(kernel_type,X,[],gamma);
        mat1 = K*L*K'+lambda*eye(m);
        mat2 = K*H*K';
        [A,~] = eigs(pinv(mat1)*mat2,dim);
        %[A,~] = eigs(mat1,mat2,dim,'SM');与上面等价
        Z = A' * X;
    end
    Z = Z * diag(sparse(1./sqrt(sum(Z.^2))));
%     Z = Z * diag(1./sqrt(sum(Z.^2)));%对每一列做二范数归一化
    Xs_new = Z(:,1:ns)';
    Xt_new = Z(:,ns+1:end)';
end

%%
function K = TCA_kernel(ker,X,X2,gamma)

    switch ker
        case 'linear'

            if isempty(X2)
                K = X'*X;
            else
                K = X'*X2;
            end

        case 'rbf'

            n1sq = sum(X.^2,1);
            n1 = size(X,2);

            if isempty(X2)
                D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            else
                n2sq = sum(X2.^2,1);
                n2 = size(X2,2);
                D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
            end
            K = exp(-gamma*D); 

        case 'sam'

            if isempty(X2)
                D = X'*X;
            else
                D = X'*X2;
            end
            K = exp(-gamma*acos(D).^2);

        otherwise
            error(['Unsupported kernel ' ker])
    end
end
