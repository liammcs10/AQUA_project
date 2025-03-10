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
e=0.05; f = 5; tau = 5;
[v2,u2,w2,st2]= aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau);
e=0.05; f = 5; tau = 0;
[v3,~,~,st3] = aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau);

disp(st1(1) - st2(1))

figure(1)
subplot(3,2,1); plot(h*(1:round(T/h))/1000, v1); xlim([.008*T/1000 .03*T/1000]), ylim([-60 30]);
subplot(3,2,3); plot(h*(1:round(T/h))/1000, v2); xlim([.008*T/1000 .03*T/1000]), ylim([-60 30]);
subplot(3,2,5); plot(h*(1:round(T/h))/1000, v3); xlim([.008*T/1000 .03*T/1000]), ylim([-60 30]);
subplot(3,2,2); plot(h*(1:round(T/h))/1000, v1); xlim([.97*T/1000 T/1000]), ylim([-60 30]);
subplot(3,2,4); plot(h*(1:round(T/h))/1000, v2); xlim([.97*T/1000 T/1000]), ylim([-60 30]);
subplot(3,2,6); plot(h*(1:round(T/h))/1000, v3); xlim([.97*T/1000 T/1000]), ylim([-60 30]);

subplot(3,2,1); title('no autapse');
subplot(3,2,3); title('delayed autapse');
subplot(3,2,5); title('instantaneous autapse'); xlabel('time [sec]'); ylabel('membrane potential [mV]');
subplot(3,2,2); title('no autapse');
subplot(3,2,4); title('delayed autapse');
subplot(3,2,6); title('instantaneous autapse');

figure(2);
subplot(1,2,1); isi2 = diff(st2); plot(isi2(1:end-1),isi2(2:end),'.');
subplot(1,2,2); isi3 = diff(st3); plot(isi3(1:end-1),isi3(2:end),'.');
% subplot(1,2,1); isi2 = diff(st2); plot(isi2(end-1000+1:end-1),isi2(end-1000+2:end),'.');
% subplot(1,2,2); isi3 = diff(st3); plot(isi3(end-1000+1:end-1),isi3(end-1000+2:end),'.');

