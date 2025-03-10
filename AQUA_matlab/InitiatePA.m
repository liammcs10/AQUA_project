%% Simulating quadratic I&F neuron with adaptation and autapse

clc
clear
close all

addpath(genpath('../HelperFunc'));
clr = defineColors;

%%% RS

C=50; vr=-60; vt=-40; k=1.5; % parameters used for IB
a=0.03; b=1; c=-40; d=10; % neocortical pyramidal neurons
vpeak=25; % spike cutoff

T=10000; h=.001; % time span and step (ms)
% I=[zeros(1,0.1*round(T/h)),500*ones(1,0.9*round(T/h))];% pulse of input DC current
% I=[zeros(1,0.1*round(T/h)),500*ones(1,0.01*round(T/h)),500*ones(1,0.89*round(T/h))];% pulse of input DC current

figure(1);
%
I=[0*ones(1,0.01*round(T/h)),158*ones(1,0.0074*round(T/h)),10*ones(1,0.9826*round(T/h))]; % pulse of input DC current
e=0.0; f = 0; tau = 0;
[v1,~,~,~] = aqua(T,h,I,vpeak,C,vr,vt,k,a,b,c,d,e,f,tau);
e=0.2; f = 100; tau = 0;
[v3,~,~,~] = aqua(T,h,I,vpeak,C,vr,vt,k,a,b,c,d,e,f,tau);
subplot(3,2,2); plot(h*(1:round(T/h))/1000, I) ;
xlim([.095 .25]); ylim([-5,165]); set(gca,'FontSize',12);
subplot(3,2,4); plot(h*(1:round(T/h))/1000, v1);
xlim([.095 .25]); set(gca,'FontSize',12);
subplot(3,2,6); plot(h*(1:round(T/h))/1000, v3); 
xlim([.095 .25]); xlabel('time [sec]'); set(gca,'FontSize',12);
%

a=0.02; b=0.2; c=-65; d=8; % RS
vpeak=30; % spike cutoff

T=500; h=.001; % time span and step (ms)
% I=[zeros(1,0.1*round(T/h)),500*ones(1,0.9*round(T/h))];% pulse of input DC current
% I=[zeros(1,0.1*round(T/h)),500*ones(1,0.01*round(T/h)),500*ones(1,0.89*round(T/h))];% pulse of input DC current

I=[zeros(1,0.1*round(T/h)),14*ones(1,0.9*round(T/h))];% pulse of input DC current
e=0.0; f = 0; tau = 0;
[v1,~,~,~] = aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau);
e=0.05; f = 6; tau = 0;
[v3,~,~,~] = aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau);
subplot(3,2,1); plot(h*(1:round(T/h))/1000, I) ; xlim([0,.25]);
ylim([min(I)-1,15]); ylabel('I [pA]'); set(gca,'FontSize',12); 
subplot(3,2,3); plot(h*(1:round(T/h))/1000, v1); xlim([0,.25]);
ylabel('V [mV]; no autapse'); set(gca,'FontSize',12);
subplot(3,2,5); plot(h*(1:round(T/h))/1000, v3);  xlim([0,.25]);
ylabel('V [mV]; with autapse');
xlabel('time [sec]'); set(gca,'FontSize',12);