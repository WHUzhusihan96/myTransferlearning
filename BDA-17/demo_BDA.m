%% Choose domains from Office+Caltech
%%% 'Caltech10', 'amazon', 'webcam', 'dslr' 
src = 'caltech.mat';
tgt = 'amazon.mat';

%% Load data
load(['data/' src]);     % source domain
fts = double(feas);
Xs = zscore(fts,1);    clear feas
Ys = label;           clear label

load(['data/' tgt]);     % target domain
fts = double(feas);
Xt = zscore(fts,1);     clear feas
Yt = label;            clear label

%% Set algorithm options
options.gamma = 1.0;
options.lambda = 0.1;
options.kernel_type = 'primal';
options.T = 10;
options.dim = 50;
options.mu = 0.5;
options.mode = 'BDA';
%% Run algorithm
[Acc,acc_ite,~] = BDA(Xs,Ys,Xt,Yt,options);
fprintf('Acc:%.2f',Acc);
