%% Simulating quadratic I&F neuron with adaptation and autapse

clc
clear
close all

addpath(genpath('../HelperFunc'));

%%% RS

a=0.02; b=0.2; c=-65; d=8; % RS
vpeak=30; % spike cutoff

T=1000; h=.01; % time span and step (ms)
% I=[zeros(1,0.1*round(T/h)),500*ones(1,0.9*round(T/h))];% pulse of input DC current
% I=[zeros(1,0.1*round(T/h)),500*ones(1,0.01*round(T/h)),500*ones(1,0.89*round(T/h))];% pulse of input DC current
%I=[zeros(1,0.1*round(T/h)),4.75*ones(1,0.9*round(T/h))];% pulse of input DC current
I = [zeros(1, 40000), 4.75*ones(1, 500), zeros(1, 10000), 5.5*ones(1, 500), zeros(1, 49000)];

e=0.0; f = 0; tau = 0;
[v1,~,~,st1] = aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau);
e=0.05; f = 5; tau = 10;
[v2,u2,w2,st2]= aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau);
e=0.05; f = 5; tau = 0;
[v3,~,~,st3] = aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau);

disp(st1(1) - st2(1))

figure(1)
subplot(1, 1, 1); 
p = plot(h*(1:round(T/h))/1000, v1, h*(1:round(T/h))/1000, v2);
p(1).LineStyle = '-';
p(1).Color = 'red';
p(1).LineWidth = 2;
p(2).LineStyle = '--';
p(2).Color = 'blue';
p(2).LineWidth = 2;
xlim([.4*T/1000 .42*T/1000]), ylim([-80 30]);
