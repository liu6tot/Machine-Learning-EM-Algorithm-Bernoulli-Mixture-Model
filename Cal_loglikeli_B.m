load 'problem2forHW4.mat'

%initial val 
[N,M]=size(dataset);
K=5;
fold=5; %cross validation fold 

indices = crossvalind('Kfold',N,fold);  %Cross Validation 
train_like = zeros(K,fold); %trainning data for clusering 
test_like = zeros(K,fold); %test data for clustering 
nCV=N/fold; %length of training set and test set. 

for i = 1:K
    for d = 1:fold 
        
        %data set
        test = (indices == d);
        train = ~test;
        train_data = dataset(train,:);
        test_data = dataset(test,:);
        
        %optimal coefficients 
        [like,ll,new_miu,new_pi] = emBernoB(train_data,i);
        train_like(i,d) = ll;
        
        %calculate test loglike
        %testset_E_step   
        taut = zeros(nCV,i);
        for k = 1:i
            for n=1:nCV
            taut(n,k) = new_pi(k,1) * new_miu(k,1) ^ sum(test_data(n,:)) * (1-new_miu(k,1))^(M - sum(test_data(n,:)));
            end
        end 
        
        %loglikelihood
        llt=0;
        lt=0;
        for n=1:nCV
            lt=sum(taut(n,:));
            llt=llt+log(lt);
        end
        
        test_like (i,d)=llt;
    end
end

%Report the training and testing log-likelihoods (average and standard deviation)
trainll_mean= mean(train_like,2); %compute mean of each row
testll_mean= mean(test_like,2);
trainll_std = std(train_like,0,2);
testll_std = std(test_like,0,2);
KName = {'1k';'2k';'3k';'4k';'5k'};
TABLE_1_train_and_test_log_likelihoods = table(trainll_mean,trainll_std,testll_mean,testll_std,'RowNames',KName);


%boxplot 
 
subplot(2,1,1);
hold on 
boxplot(test_like');
plot(testll_mean,'r--','Linewidth',2);
title('Figure 2 : Test log likeligood');
hold

subplot(2,1,2);
hold on 
boxplot(train_like');
plot(trainll_mean,'k--','Linewidth',2);
title('Train log likeligood');
hold


%Plot Best Model 
[~,bestModel] = max(testll_mean); %identify the best model 
% Figure 
figure (2);
hold on; 
plot(trainll_mean,'k','Linewidth',2);
plot(testll_mean,'r','Linewidth',2);
yl = ylim;
plot([bestModel,bestModel],[yl(1),yl(2)],'k--');
xlim([1,K]);
xlabel('Number of Clusters')
ylabel ('Log_Likelihoods')
legend('train log','test log','Best Model')
hold off;



[like,ll,new_miu,new_pi] = emBernoB(dataset,bestModel);


%reporting result of new_pi and new_miu 
k_val=zeros(bestModel,1);
for i = 1:bestModel
    k_val(i,1)=i;
end 
TABLE_2_Best_K_Optimal_Coefficients = table(k_val,new_miu,new_pi);

miu_mean=mean(new_miu);
miu_std=std(new_miu); 
pi_mean=mean(new_pi); 
pi_std=std(new_pi);
TABLE_3_Best_K_Optimal_Coefficients_Average_and_Std = table(miu_mean, miu_std,pi_mean,pi_std);

%plot new_ pi and new_miu
%new_miu boxplot
figure(3);
subplot(1,2,1);
boxplot(new_miu);
title('BestK Mean Boxplot');
%new_pi boxplot 
subplot(1,2,2);
boxplot(new_pi);
title('BestK Pi Boxplot');

TABLE_1_train_and_test_log_likelihoods
TABLE_2_Best_K_Optimal_Coefficients
TABLE_3_Best_K_Optimal_Coefficients_Average_and_Std
