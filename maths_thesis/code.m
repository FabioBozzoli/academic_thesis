clear
close all
load 'dataset_OCR.mat'

f = zeros(36,1000);
for i = 1:1000
    f(:,i) = computeBRISQUEFeatures(reshape(IM_TRAIN(:,i),365,590));
end
%% 

tic;
svrModel = fitrsvm(f',LABEL_TRAIN,"KernelFunction","rbf", "Standardize",true,"KernelScale","auto","BoxConstraint",10000);

time_train_svr = toc;

tic;
rfModel = fitrensemble(f',LABEL_TRAIN,"Method","Bag","Learners","tree","NumLearningCycles",1);
time_train_rf = toc;

g = zeros(36,230);
for i = 1:230
    g(:,i) = computeBRISQUEFeatures(reshape(IM_TEST(:,i),365,590));
end

y_predict_svr_train = predict(svrModel, f');
y_predict_svr_test = predict(svrModel, g');

y_predict_rf_train = predict(rfModel, f');
y_predict_rf_test = predict(rfModel, g');

%Calcolo dell'R^2 train
SS_tot = sum((LABEL_TRAIN - mean(LABEL_TRAIN)).^2);
SS_res_svr = sum((LABEL_TRAIN - y_predict_svr_train).^2);
SS_res_rf = sum((LABEL_TRAIN - y_predict_rf_train).^2);
R2_svr_train = 1-(SS_res_svr/SS_tot);
R2_rf_train = 1-(SS_res_rf/SS_tot);

%Calcolo dell'R^2 test
SS_tot = sum((LABEL_TEST - mean(LABEL_TEST)).^2);
SS_res_svr = sum((LABEL_TEST - y_predict_svr_test).^2);
SS_res_rf = sum((LABEL_TEST - y_predict_rf_test).^2);
R2_svr_test = 1-(SS_res_svr/SS_tot);
R2_rf_test = 1-(SS_res_rf/SS_tot);

%Mean absolute error train
MAE_svr_train = mean(abs(LABEL_TRAIN - y_predict_svr_train));
MAE_rf_train = mean(abs(LABEL_TRAIN - y_predict_rf_train));

%Mean absolute error test
MAE_svr_test = mean(abs(LABEL_TEST - y_predict_svr_test));
MAE_rf_test = mean(abs(LABEL_TEST - y_predict_rf_test));

%figure;
%plot(1:20, LABEL_TRAIN(1:20), '*b', 'LineWidth', 2);
%hold on;  
%plot(1:20, y_predict_svr_train(1:20), '*r', 'LineWidth', 2);  
%hold on;  
%plot(1:20, y_predict_rf(1:20), '*g', 'LineWidth', 2); 
%hold off;  
%legend({'ground truth', 'SVR'});


