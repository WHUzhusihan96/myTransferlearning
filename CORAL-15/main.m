clear;
%��ȡԴ������#SUN->S1 Scene-15->S2 banjaluka->T1, UCMERCED->T2  C19->T3
%caltech,amazon,webcam,dslr
load('/data/dslr.mat');
src_data = feas;src_label = label;clear data;clear label;
% src_data = zscore(double(src_data));
%��ȡĿ��������
load('/data/webcam.mat');
tar_data = feas;tar_label = label;clear data;clear label;
% tar_data = zscore(double(tar_data));
Xs_new = CORAL(src_data,tar_data);

% knn_model = fitcknn(Xs_new,src_label,'NumNeighbors',1);
% y_pred = knn_model.predict(tar_data);
% acc = length(find(y_pred==tar_label))/length(tar_label);

% [acc,~,time_pass] = CORAL_NN(src_data,src_label,tar_data,tar_label);

% model = libsvmtrain(src_label, Xs_new,'-c 100');
% [pred, acc, ~] = libsvmpredict(tar_label, tar_data, model);

% [acc,~,time_pass] = CORAL_SVM(src_data,src_label,tar_data,tar_label);

%����ʵ�飬CORAL������������ʹ��SVM����������ò���SIM����ʹ��NN����������ֱ��ʹ�ã��������ȸ�
%Ҳ��������ĵ�1���͵�4������������SA�ȷ���Ҳ�����Ƶġ�
%��CORAL��������û�����ƽ��ͣ�SA���н��ͣ��������෴�ģ�We use SIM directly to perform a ?nearest neighbor classification task. 
%On the other hand, since SIM not PSD we can not make use of it to learna SVM directly.
