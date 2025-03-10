%% Adaptive QUadratic with Autapse (AQUA) IF model

function [v,u,w,st] = aqua2(T,h,I,vpeak,a,b,c,d,e,f,tau,p)

if nargin < 9
  e = 0; f = 0; tau = 0; p = 0;
elseif nargin < 11
  tau = 0; p = 0;
elseif nargin < 12
  p = 0;
end

N=round(T/h); % number of simulation steps
v=-80*ones(N,1); u=0*v; w=u; st=nan*u; % initial values

% fnc = {@(v,u,w,I) (.04*v.*v + 5*v + 140 - u + w + I);
%      @(v,u)     (a*(b*v - u));
%      @(w)       (-e*w)};

for t = 1+round(tau/h):N-1 % forward Euler method
%   k1 = h*[fnc{1}(v(t),u(t),w(t-round(tau/h)),I(t));
%           fnc{2}(v(t),u(t));
%           fnc{3}(w(t))];
%   k2 = h*[f{1}(v(t)+k1(1)/2,u(t)+k1(2)/2,w(t-round(tau/h))+k1(3)/2,I(t));
%           f{2}(v(t),u(t));
%           f{3}(w(t))];
%   v(t+1)=v(t)+k1(1);
%   u(t+1)=u(t)+k1(2);
%   w(t+1)=w(t)+k1(3);

  v(t+1)=v(t)+h*(.04*v(t)*v(t) + 5*v(t) + 140 - u(t) + w(t-round(tau/h)) + I(t));
  u(t+1)=u(t)+h*a*(b*v(t)-u(t));
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