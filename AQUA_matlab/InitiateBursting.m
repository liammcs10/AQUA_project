%% Simulating quadratic I&F neuron with adaptation and autapse

clc
clear
close all

addpath(genpath('../HelperFunc'));
clr = defineColors;

%%% RS

a=0.02; b=0.2; c=-65; d=8; % RS
vpeak=30; % spike cutoff

T=500; h=.001; % time span and step (ms)
% I=[zeros(1,0.1*round(T/h)),500*ones(1,0.9*round(T/h))];% pulse of input DC current
% I=[zeros(1,0.1*round(T/h)),500*ones(1,0.01*round(T/h)),500*ones(1,0.89*round(T/h))];% pulse of input DC current

figure('Units','normalized','OuterPosition',[0 0 1 1]);
%
I=[zeros(1,0.1*round(T/h)),5*ones(1,0.9*round(T/h))];% pulse of input DC current
e=0.0; f = 0; tau = 0;
[v1,~,~,~] = aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau);
e=0.05; f = 6; tau = 0;
[v3,~,~,~] = aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau);
subplot(3,4,1); plot(h*(1:round(T/h))/1000, I) ; ylim([min(I)-1,15]); ylabel('I [pA]'); set(gca,'FontSize',12);
subplot(3,4,5); plot(h*(1:round(T/h))/1000, v1);                      ylabel('V [mV]; no autapse'); set(gca,'FontSize',12);
subplot(3,4,9); plot(h*(1:round(T/h))/1000, v3);                      ylabel('V [mV]; with autapse');
xlabel('time [sec]'); set(gca,'FontSize',12);
%
I=[zeros(1,0.1*round(T/h)),8*ones(1,0.9*round(T/h))];% pulse of input DC current
e=0.0; f = 0; tau = 0;
[v1,~,~,~] = aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau);
e=0.05; f = 6; tau = 0;
[v3,~,~,~] = aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau);
subplot(3,4,2); plot(h*(1:round(T/h))/1000, I) ; ylim([min(I)-1,15]); ylabel('I [pA]'); set(gca,'FontSize',12);
subplot(3,4,6); plot(h*(1:round(T/h))/1000, v1);                      ylabel('V [mV]; no autapse'); set(gca,'FontSize',12);
subplot(3,4,10); plot(h*(1:round(T/h))/1000, v3);                      ylabel('V [mV]; with autapse');
xlabel('time [sec]'); set(gca,'FontSize',12);
%
I=[zeros(1,0.1*round(T/h)),11*ones(1,0.9*round(T/h))];% pulse of input DC current
e=0.0; f = 0; tau = 0;
[v1,~,~,~] = aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau);
e=0.05; f = 6; tau = 0;
[v3,~,~,~] = aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau);
subplot(3,4,3); plot(h*(1:round(T/h))/1000, I) ; ylim([min(I)-1,15]); ylabel('I [pA]'); set(gca,'FontSize',12);
subplot(3,4,7); plot(h*(1:round(T/h))/1000, v1);                      ylabel('V [mV]; no autapse'); set(gca,'FontSize',12);
subplot(3,4,11); plot(h*(1:round(T/h))/1000, v3);                      ylabel('V [mV]; with autapse');
xlabel('time [sec]'); set(gca,'FontSize',12);
%
I=[zeros(1,0.1*round(T/h)),14*ones(1,0.9*round(T/h))];% pulse of input DC current
e=0.0; f = 0; tau = 0;
[v1,~,~,~] = aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau);
e=0.05; f = 8; tau = 0;
[v3,~,~,~] = aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau);
subplot(3,4,4); plot(h*(1:round(T/h))/1000, I) ; ylim([min(I)-1,15]); ylabel('I [pA]'); set(gca,'FontSize',12);
subplot(3,4,8); plot(h*(1:round(T/h))/1000, v1);                      ylabel('V [mV]; no autapse'); set(gca,'FontSize',12);
subplot(3,4,12); plot(h*(1:round(T/h))/1000, v3,'Color',clr(2,:));    ylabel('V [mV]; with autapse');
e=0.05; f = 6; tau = 0;
[v3,~,~,~] = aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau);
hold on; plot(h*(1:round(T/h))/1000, v3,'Color',clr(1,:));                      ylabel('V [mV]; with autapse');
xlabel('time [sec]'); set(gca,'FontSize',12);