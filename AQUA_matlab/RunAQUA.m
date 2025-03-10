%% Simulating quadratic I&F neuron with adaptation and autapse

clc
clear
close all

addpath(genpath('../HelperFunc'));

%%% bursting

% C=50; vr=-60; vt=-40; k=1.5; % parameters used for IB
% a=0.03; b=1; c=-40; d=150; % neocortical pyramidal neurons
% vpeak=25; % spike cutoff
% % vpeak=vt; % spike cutoff
% 
% T=50000; h=.001; % time span and step (ms)
% % I=[zeros(1,0.1*round(T/h)),500*ones(1,0.9*round(T/h))];% pulse of input DC current
% % I=[zeros(1,0.1*round(T/h)),500*ones(1,0.01*round(T/h)),500*ones(1,0.89*round(T/h))];% pulse of input DC current
% I=[zeros(1,0.01*round(T/h)),680*ones(1,0.99*round(T/h))];% pulse of input DC current
% 
% e=0.0; f = 0; tau = 0;
% [v1,~,~,~] = aqua(T,h,I,vpeak,C,vr,vt,k,a,b,c,d,e,f,tau);
% e=0.2; f = 100; tau = 5;
% [v2,~,~,st2]= aqua(T,h,I,vpeak,C,vr,vt,k,a,b,c,d,e,f,tau);
% e=0.2; f = 100; tau = 0;
% [v3,~,~,st3] = aqua(T,h,I,vpeak,C,vr,vt,k,a,b,c,d,e,f,tau);
% 
% figure(1)
% subplot(3,2,1); plot(h*(1:round(T/h))/1000, v1); xlim([.008*T/1000 .03*T/1000]); ylim([-60 -35]);
% subplot(3,2,3); plot(h*(1:round(T/h))/1000, v2); xlim([.008*T/1000 .03*T/1000]); ylim([-60 -35]);
% subplot(3,2,5); plot(h*(1:round(T/h))/1000, v3); xlim([.008*T/1000 .03*T/1000]); ylim([-60 -35]);
% subplot(3,2,2); plot(h*(1:round(T/h))/1000, v1); xlim([.97*T/1000 T/1000]); ylim([-60 -35]);
% subplot(3,2,4); plot(h*(1:round(T/h))/1000, v2); xlim([.97*T/1000 T/1000]); ylim([-60 -35]);
% subplot(3,2,6); plot(h*(1:round(T/h))/1000, v3); xlim([.97*T/1000 T/1000]); ylim([-60 -35]);
% 
% subplot(3,2,1); title('no autapse');
% subplot(3,2,3); title('delayed autapse');
% subplot(3,2,5); title('instantaneous autapse'); xlabel('time [sec]'); ylabel('membrane potential [mV]');
% subplot(3,2,2); title('no autapse');
% subplot(3,2,4); title('delayed autapse');
% subplot(3,2,6); title('instantaneous autapse');
% 
% figure(2);
% subplot(1,2,1); isi2 = diff(st2); plot(isi2(end-1000+1:end-1),isi2(end-1000+2:end),'.');
% subplot(1,2,2); isi3 = diff(st3); plot(isi3(end-1000+1:end-1),isi3(end-1000+2:end),'.');

% 
% e=0.0; f = 0; tau = 0;
% [v1,~,~,~] = aqua(T,h,I,vpeak,C,vr,vt,k,a,b,c,d,e,f,tau,2);
% e=0.2; f = 100; tau = 5;
% [v22,~,~,st22]= aqua(T,h,I,vpeak,C,vr,vt,k,a,b,c,d,e,f,tau,2);
% e=0.2; f = 100; tau = 0;
% [v3,~,~,st32] = aqua(T,h,I,vpeak,C,vr,vt,k,a,b,c,d,e,f,tau,2);
% 
% figure(2);
% subplot(1,2,1); hold on; isi22 = diff(st22); plot(isi22(end-1000+1:end-1),isi22(end-1000+2:end),'.');
% subplot(1,2,2); hold on; isi32 = diff(st32); plot(isi32(end-1000+1:end-1),isi32(end-1000+2:end),'.');
% 
% figure;plot(h*(9000000+(1:100000)-10001)/1000,v2(9000000+(1:100000)-10001));
% hold on;plot(h*(9000000+(1:100000)-10001)/1000,v22(9000000+(1:100000)-10001));
% xlabel('time [sec]'); ylabel('[mV]');
% title('membrane potential diverges exponentially due to perturbation');

%%% persistent

C=50; vr=-60; vt=-40; k=1.5; % parameters used for IB
a=0.03; b=1; c=-40; d=10; % neocortical pyramidal neurons
vpeak=25; % spike cutoff
% vpeak=vt; % spike cutoff

T=10000; h=.001; % time span and step (ms)
% I=[zeros(1,0.1*round(T/h)),500*ones(1,0.9*round(T/h))];% pulse of input DC current
I=[0*ones(1,0.01*round(T/h)),158*ones(1,0.0074*round(T/h)),10*ones(1,0.9826*round(T/h))];
% I=[zeros(1,0.01*round(T/h)),680*ones(1,0.99*round(T/h))];% pulse of input DC current

e=0.0; f = 0; tau = 0;
[v1,~,~,~] = aqua(T,h,I,vpeak,C,vr,vt,k,a,b,c,d,e,f,tau);
e=0.2; f = 100; tau = 5; % delayed autapse
[v2,~,~,st2]= aqua(T,h,I,vpeak,C,vr,vt,k,a,b,c,d,e,f,tau);
e=0.2; f = 100; tau = 0; % instantaneous autapse
[v3,~,~,st3] = aqua(T,h,I,vpeak,C,vr,vt,k,a,b,c,d,e,f,tau);

figure(3)
subplot(3,2,1); plot(h*(1:round(T/h))/1000, v1); xlim([.008*T/1000 .03*T/1000]); ylim([-60 30]);
subplot(3,2,3); plot(h*(1:round(T/h))/1000, v2); xlim([.008*T/1000 .03*T/1000]); ylim([-60 30]);
subplot(3,2,5); plot(h*(1:round(T/h))/1000, v3); xlim([.008*T/1000 .03*T/1000]); ylim([-60 30]);
subplot(3,2,2); plot(h*(1:round(T/h))/1000, v1); xlim([.97*T/1000 T/1000]); ylim([-60 30]);
subplot(3,2,4); plot(h*(1:round(T/h))/1000, v2); xlim([.97*T/1000 T/1000]); ylim([-60 30]);
subplot(3,2,6); plot(h*(1:round(T/h))/1000, v3); xlim([.97*T/1000 T/1000]); ylim([-60 30]);

subplot(3,2,1); title('no autapse');
subplot(3,2,3); title('delayed autapse');
subplot(3,2,5); title('instantaneous autapse'); xlabel('time [sec]'); ylabel('membrane potential [mV]');
subplot(3,2,2); title('no autapse');
subplot(3,2,4); title('delayed autapse');
subplot(3,2,6); title('instantaneous autapse');

figure(4);
subplot(1,2,1); isi2 = diff(st2); plot(isi2(end-1000+1:end-1),isi2(end-1000+2:end),'.');
subplot(1,2,2); isi3 = diff(st3); plot(isi3(end-1000+1:end-1),isi3(end-1000+2:end),'.');

% [x,y,z] = lorenz(T,h,1,1,1,10,28,8/3);
% figure;plot3(x,y,z);

