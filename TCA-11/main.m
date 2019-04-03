clear;
%读取源域数据#SUN->S1 Scene-15->S2 banjaluka->T1, UCMERCED->T2  C19->T3
%caltech,amazon,webcam,dslr
load('/data/dslr.mat');
src_data =  feas; src_label = label;
%读取目标域数据
load('/data/caltech.mat');
tar_data =  feas; tar_label = label;
%设置参数kernal可以是rbf,linear,sam,primal, 
options.dim = 30;options.lambda = 1;
options.gamma =1 ;options.kernel_type = 'primal';
[X_src_new,X_tar_new,A] = myTCA(src_data,tar_data,options);
% model = libsvmtrain(src_label, X_src_new,'-c 100');
% [pred, acc, ~] = libsvmpredict(tar_label, X_tar_new, model);
knn_model = fitcknn(X_src_new,src_label,'NumNeighbors',1);%Xs_new,Ys
Y_tar_pseudo = knn_model.predict(X_tar_new);%Xt_new
acc = length(find(Y_tar_pseudo==tar_label))/length(tar_label)
%运行KNN分类器
