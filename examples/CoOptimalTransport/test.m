clear;
close all;

rng(101)

%% load sample USPS - MNIST data
mydata = load('Data_MNIST_USPS_prop50.mat');
S = mydata.Y;
T = mydata.X;

lambdaOT1 = 5e-4; % sample - sample entropy regularizer.
lambdaOT2 = 5e-4; % feature - feature entropy regularizer.
maxiter = 30;



% Using MOT
options.lambda_samples = lambdaOT1;
options.lambda_features = lambdaOT2;
options.maxiter = maxiter;
options.method ='CG';

[C1, gamma1, W1, infos1] = COT_with_MOT(S, T, options);



% Alternating approach
[C2, gamma2, W2, infos2] = COT(S, T, maxiter, lambdaOT1, lambdaOT2);




% Plots


figure(101);
plot([infos1.cost],'r-','Linewidth',2);  
hold on; 
plot(infos2.cost,'b--','Linewidth',2);
hold off;
xlabel('Iterations','fontsize',20,'fontweight','bold');
ylabel('Cost','fontsize',20,'fontweight','bold');      
legend({'MOT','Alternating'},'Location','northeast')
set(gca,'fontsize',20,'fontweight','bold');
set(gcf,'color','w');
saveas(gcf,'MOT_vs_Alternating_cost.pdf')




figure(101);
plot([infos1.time], [infos1.cost],'r-','Linewidth',2);  
hold on; 
plot(infos2.time, infos2.cost,'b--','Linewidth',2);
hold off;
xlabel('Time','fontsize',20,'fontweight','bold');
ylabel('Cost','fontsize',20,'fontweight','bold');      
legend({'MOT','Alternating'},'Location','northeast')
set(gca,'fontsize',20,'fontweight','bold');
set(gcf,'color','w');
saveas(gcf,'MOT_vs_Alternating_time.pdf')


