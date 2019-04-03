clear;
%caltech,amazon,webcam,dslr
load('/data/dslr.mat');
src_data =  feas; src_label = label;
%读取目标域数据
load('/data/webcam.mat');
tar_data =  feas; tar_label = label;
%设置参数kernal可以是rbf,linear,sam,primal, 
options.dim = 30;options.lambda = 1;
options.gamma =1 ;options.kernel_type = 'primal';options.T = 10;
[~,acc_ite,A] = myJDA(src_data,src_label,tar_data,tar_label,options);
