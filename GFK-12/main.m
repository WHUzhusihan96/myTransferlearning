clear;
%caltech,amazon,webcam,dslr
load('/data/caltech.mat');
src_data = feas;src_label = label;clear data;clear label;
%读取目标域数据
load('/data/amazon.mat');
tar_data = feas;tar_label = label;clear data;clear label;
acc_i = [];
for i = 20 : 100
    [acc,G,Cls] = GFK(src_data,src_label,tar_data,tar_label,i);
    acc_i = [acc_i;acc];
end
%运行SVM进行分类
%model = libsvmtrain(src_label, X_src_new,'-c 100');
%[pred, acc, ~] = libsvmpredict(tar_label, X_tar_new, model);
%运行KNN分类器
