clear;
%��ȡԴ������#SUN->S1 Scene-15->S2 banjaluka->T1, UCMERCED->T2  C19->T3
%caltech,amazon,webcam,dslr
load('/data/dslr.mat');
src_data =  feas; src_label = label;
%��ȡĿ��������
load('/data/webcam.mat');
tar_data =  feas; tar_label = label;
%���ò���kernal������rbf,linear,sam,primal, 
options.dim = 30;options.lambda = 1;
options.gamma =1 ;options.kernel_type = 'primal';options.T = 10;
[~,acc_ite,A] = myJDA(src_data,src_label,tar_data,tar_label,options);