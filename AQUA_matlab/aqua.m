%% Adaptive QUadratic with Autapse (AQUA) IF model

function [v,u,w,st] = aqua(T,h,I,vpeak,C,vr,vt,k,a,b,c,d,e,f,tau,p)

if nargin < 13
  e = 0; f = 0; tau = 0; p = 0;
elseif nargin < 15
  tau = 0; p = 0;
elseif nargin < 16
  p = 0;
end

N=round(T/h); % number of simulation steps
v=vr*ones(N,1); u=0*v; w=u; st=nan*u; % initial values

% for t = 1:N-1 % forward Euler method
%   v(t+1)=v(t)+h*(k*(v(t)-vr)*(v(t)-vt)-u(t)+w(t)+I(t))/C;
%   u(t+1)=u(t)+h*a*(b*(v(t)-vr)-u(t));
%   if t+1+round(tau/h)<=N
%     w(t+1+round(tau/h))=w(t+round(tau/h))-h*e*w(t+round(tau/h));
%   end
%   if v(t+1)>=vpeak % a spike is fired!
%     v(t)=vpeak; % padding the spike amplitude
%     v(t+1)=c; % membrane voltage reset
%     u(t+1)=u(t+1)+d; % recovery variable update
%     if t+1+round(tau/h)<=N
%       w(t+1+round(tau/h))=w(t+1+round(tau/h))+f; % autapse current update
%     end
%     st(find(isnan(st),1)) = t*h;
%   end
% end
for t = 1+round(tau/h):N-1 % forward Euler method
  v(t+1)=v(t)+h*(k*(v(t)-vr)*(v(t)-vt)-u(t)+w(t-round(tau/h))+I(t))/C;
  u(t+1)=u(t)+h*a*(b*(v(t)-vr)-u(t));
  w(t+1)=w(t)-h*e*w(t);
  if v(t+1)>=vpeak % a spike is fired!
    v(t)=vpeak; % padding the spike amplitude
    v(t+1)=c; % membrane voltage reset
    u(t+1)=u(t+1)+d; % recovery variable update
    w(t+1)=w(t+1)+f; % autapse current update
    st(find(isnan(st),1)) = t*h;
  end
  if t == round(.9*N)
    v(t+1) = v(t+1) + p;
  end
end
st(isnan(st)) = [];