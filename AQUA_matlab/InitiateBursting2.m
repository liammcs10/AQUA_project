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

figure;

I=[zeros(1,0.1*round(T/h)),14*ones(1,0.9*round(T/h))];% pulse of input DC current
e=0.0; f = 0; tau = 0;
[v1,~,~,~] = aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau);
e=0.05; f = 6; tau = 0;
[v3,~,~,~] = aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau);
subplot(3,1,1); plot(h*(1:round(T/h))/1000, I) ; xlim([0,.2]);
ylim([min(I)-1,15]); ylabel('I [pA]'); set(gca,'FontSize',12); 
subplot(3,1,2); plot(h*(1:round(T/h))/1000, v1); xlim([0,.2]);
ylabel('V [mV]; no autapse'); set(gca,'FontSize',12);
subplot(3,1,3); plot(h*(1:round(T/h))/1000, v3);  xlim([0,.2]);
ylabel('V [mV]; with autapse');
xlabel('time [sec]'); set(gca,'FontSize',12);
