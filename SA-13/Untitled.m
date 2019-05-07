clear;
%读取源域数据#SUN->S1 Scene-15->S2 banjaluka->T1, UCMERCED->T2  C19->T3
%caltech,amazon,webcam,dslr
load('/data/caltech.mat');
src_data =  feas; src_label = label;
src_data =  zscore(double(src_data),1);
%读取目标域数据
load('/data/amazon.mat');
tar_data =  feas; tar_label = label;
tar_data =  zscore(double(tar_data),1);

[Xss,~,~] = pca(src_data);
[Xtt,~,~] = pca(tar_data);
Xs = Xss(:,1:30);
Xt = Xtt(:,1:30);
[acc,y_pred,time_pass] =  SA_SVM(src_data,src_label,tar_data,tar_label,Xs,Xt);