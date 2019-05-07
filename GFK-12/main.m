clear;
%读取源域数据#SUN->S1 Scene-15->S2 banjaluka->T1, UCMERCED->T2  C19->T3
%caltech,amazon,webcam,dslr
load('/newdata28/Scene-15.mat');
src_data = data;src_label = label;clear data;clear label;
src_data = zscore(double(src_data));
%读取目标域数据
load('/newdata28/C19.mat');
tar_data = data;tar_label = label;clear data;clear label;
tar_data = zscore(double(tar_data));
acc_i = [];
for i = 20 : 100
    [acc,G,Cls] = GFK(src_data,src_label,tar_data,tar_label,i);
    acc_i = [acc_i;acc];
end
%运行SVM进行分类
%model = libsvmtrain(src_label, X_src_new,'-c 100');
%[pred, acc, ~] = libsvmpredict(tar_label, X_tar_new, model);
%运行KNN分类器
