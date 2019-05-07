clear;
%caltech,amazon,webcam,dslr
load('/data/dslr.mat');
src_data = feas;src_label = label;clear data;clear label;
%读取目标域数据
load('/data/webcam.mat');
tar_data = feas;tar_label = label;clear data;clear label;
Xs_new = CORAL(src_data,tar_data);

% knn_model = fitcknn(Xs_new,src_label,'NumNeighbors',1);
% y_pred = knn_model.predict(tar_data);
% acc = length(find(y_pred==tar_label))/length(tar_label);

% [acc,~,time_pass] = CORAL_NN(src_data,src_label,tar_data,tar_label);

% model = libsvmtrain(src_label, Xs_new,'-c 100');
% [pred, acc, ~] = libsvmpredict(tar_label, tar_data, model);

% [acc,~,time_pass] = CORAL_SVM(src_data,src_label,tar_data,tar_label);

%经过实验，CORAL这个方法，如果使用SVM分类器，最好采用SIM；而使用NN分类器，则直接使用，这样精度高
%也就是上面的第1个和第4个方法。包括SA等方法也是类似的。
%在CORAL的文章中没有类似解释，SA中有解释，但是是相反的：We use SIM directly to perform a ?nearest neighbor classification task. 
%On the other hand, since SIM not PSD we can not make use of it to learna SVM directly.
