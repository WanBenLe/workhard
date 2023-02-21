function [t_forcast,new_para] = EMupf(para,y,t,dy,dt)
y=y(end);
t=t(end);
dy=dy(end);
dt=dt(end);

%给定产品数据更新单产品参数并预测
a=para(1);
b=para(2);
c=para(3);
d=para(4);
N=length(y);
a1=N/2+a;
p3_a=0;
p3_b=0;
p2=0;
for k=1:N
    p2=p2+dy(k)^2/(2*lambda(dt(k)));
    p3_a=p3_a+dy(k);
    p3_b=p3_b+lambda(dt(k));
end
p3=(c+d*sum(p3_a))^2/(2*(d+d^2* sum(p3_b)));
p1=b+c^2/(2*d);

if ~isnan(p1+p2-p3)
    b1=p1+p2-p3;
else
    b1=b;
end

c1=(sum(p3_a)*d+c)/(sum(p3_b)*d+1);
d1=d/(sum(p3_b)*d+1);




%gamma分布均值倒数是sigma^2,N分布均值c是mu
sigma2=b1/a1;
mu=c1;
l=8;
%寻找概率=0.5的值
fun = @(tx) abs(RLpdf(tx,l,sigma2,mu)-0.5);
t_forcast = fminbnd(fun,0,5000);
new_para=[a1,b1,c1,d1];
end




function x=RLpdf(tx,l,sigma2,mu)
   p1=l/(2*pi*sigma2*lambda(tx).^3)^0.5;
   p2=exp(-((l-mu*lambda(tx))^2/(2*sigma2*lambda(tx))));
   x=p1*p2;
end

function x=lambda(t)
x=t^1;
end
