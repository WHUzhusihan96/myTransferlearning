str_domains = {'caltech.mat', 'amazon', 'webcam', 'dslr'};
list_acc = [];
for i = 1 : 4
    for j = 1 : 4
        if i == j
            continue;
        end
        src = str_domains{i};
        tgt = str_domains{j};
        load(['data/' src]);     % source domain
        %fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        %Xs = zscore(fts,1);    clear fts
        Xs = feas; clear feas
        Ys = label; clear label
        
        load(['data/' tgt]);     % target domain
        %fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        %Xt = zscore(fts,1);     clear fts
        Xt = feas; clear feas
        Yt = label; clear label
        
        % BDA
        options.gamma = 1.0;
        options.lambda = 1.0;
        options.kernel_type = 'primal';
        options.T = 10;
        options.dim = 30;
        options.mu = 0.5;
        options.mode = 'W-BDA';
        %% Run algorithm
        [Acc,acc_ite,~] = BDA(Xs,Ys,Xt,Yt,options);
        list_acc = [list_acc,max(acc_ite*100)];
        %fprintf('Acc:%.2f',Acc);;
    end
end
