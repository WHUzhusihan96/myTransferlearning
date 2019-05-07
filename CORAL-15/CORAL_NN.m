function [acc,y_pred,time_pass] = CORAL_NN(Xs,Ys,Xt,Yt)
%% This combines CORAL and SVM. Very simple, very easy to use.
%% Please download libsvm and add it to Matlab path before using SVM.

    time_start = clock();
    %CORAL
    Xs = double(Xs);
    Xt = double(Xt);
    Ys = double(Ys);
    Yt = double(Yt);
    cov_source = cov(Xs) + eye(size(Xs, 2));
    cov_target = cov(Xt) + eye(size(Xt, 2));
    A_coral = cov_source^(-1/2)*cov_target^(1/2);
    Sim_coral = double(Xs * A_coral * Xt');
    [acc,y_pred] = NN_Accuracy(double(Xs), A_coral, double(Yt), Sim_coral, double(Ys));
    time_end = clock();
    time_pass = etime(time_end,time_start);
end

function [acc,y_pred] = NN_Accuracy (trainset, M,testlabelsref,Sim,trainlabels)
    % Using Libsvm
    Sim_Trn = trainset * M *  trainset';
    index = [1:1:size(Sim,1)]';
    Sim = [[1:1:size(Sim,2)]' Sim'];
    Sim_Trn = [index Sim_Trn ];
    knn_model = fitcknn(Sim_Trn,trainlabels,'NumNeighbors',1);%Xs_new,Ys
    y_pred = knn_model.predict(Sim);%Xt_new
    acc = length(find(y_pred==testlabelsref))/length(testlabelsref);
end