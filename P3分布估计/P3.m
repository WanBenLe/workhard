clc
clear
format SHORT G
px=[0.01,1,10,20,30,50,60,80,90,99.9];
phip=[9.51,3.89,1.23,0.5,0.09,-0.37,-0.51,-0.7,-0.748,-0.769];
max_data1=readmatrix('llPP.csv');
%hold on
%plot(max_data1(:,4),max_data1(:,1));
ll=readmatrix('ll.csv');
max_data=sort(ll(:,2));

disp('正态分布矩估计')
[xbar,var] = moment_norm(max_data);
disp([xbar,var])
x = 0:1:7000;
p = cdf(makedist('Normal','mu',xbar,'sigma',var),x);
figure
plot(x,p)
disp(cdfnorm(p,x))

disp('正态分布ML估计')
[xbar,var]=normfit(max_data);
disp([xbar,var])
x = 0:1:7000;
p = cdf(makedist('Normal','mu',xbar,'sigma',var),x);
figure
plot(x,p)
disp(cdfnorm(p,x))


disp('对数正态分布矩估计')
[xbar,var] = moment_norm(log(max_data));
disp([xbar,var])
x = 0:.05:8.5;
p = cdf(makedist('Normal','mu',xbar,'sigma',var),x);
x=exp(x);
figure
plot(x,p)
disp(cdfnorm(p,x))

disp('对数正态分布ML估计')
[xbar,var]=normfit(log(max_data));
disp([xbar,var])
x = 0:.05:8.5;
p = cdf(makedist('Normal','mu',xbar,'sigma',var),x);
x=exp(x);
figure
plot(x,p)
disp(cdfnorm(p,x))


disp('矩')
[alpha , beta , a0,EX,cv,cs] = fitP31( max_data );
disp(['aplha,beta,a0:',num2str([alpha , beta , a0])])
pdfp=P3pdf( max_data , alpha , beta , a0);
x = cdfcalc( phip , EX,cv );
hold on
plot(max_data,pdfp)
disp([px;x])


disp('MSE')
%disp( sum((Xcdf1-max_data1(:,4)).^2))
disp('权函数')
[alpha , beta , a0,EX,cv,cs]  = fitP32( max_data );
disp(['aplha,beta,a0:',num2str([alpha , beta , a0])])
Xcdf2 = P3cdf( max_data , alpha , beta , a0 );
pdfp=P3pdf( max_data , alpha , beta , a0);
x = cdfcalc( phip , EX,cv );
hold on
plot(max_data,pdfp)
disp([px;x])


disp('线性矩')
[alpha , beta , a0,EX,cv,cs] = fitP33( max_data1(:,1),max_data1(:,2) );
disp(['aplha,beta,a0:',num2str([alpha , beta , a0])])
Xcdf3 = P3cdf( max_data , alpha , beta , a0 );
pdfp=P3pdf( max_data , alpha , beta , a0);
%这就是流量计算,负的置0
x = cdfcalc( phip , EX,cv );
hold on
%参数估计都是对的,plot的图怪怪的
plot(max_data,pdfp)
legend('矩','权函数','线性矩')
disp([px;x])


ll=readmatrix('ll.csv');
[f,xi] = ksdensity(ll(:,2)); 
figure
plot(xi,f);

%kde
[f,xi,bw] = ksdensity(ll(:,2),'Support','positive',...
	'Function','cdf', 'NumPoints',1000);
plot(xi,f,'-g','LineWidth',2)
%流量从这里找
a=[xi',f'];

%2018年的
[F, XI,a,b]=PDFAnalyze(ll(:,2), 'plotType', 'pdf');
plot(F, XI,'-g','LineWidth',2)
plot(XI,a)
a=[xi',a'];

disp(1)


function [alpha , beta , a0,I1,cv,cs] = fitP33( X,freq )
% alpha     形状参数
% beta      尺度参数
% a0         位置参数

n = length(X);
freq=freq;
M1=mean(X);
M2=mean(X.*freq);
M3=mean(X.*freq.*freq);
I1=M1;
I2=2*M2-M1;
I3=6*M3-6*M2+M1;

cv=I2/I1;
cs=I3/I2;
alpha=4/cs^2;
beta=2/(cs*cv*I1);
a0=I1*(1-2*cv/cs);
disp(['EX,cv,cs',num2str([I1,cv,cs])])
end



function [alpha , beta , a0,meanX,cv,cs] = fitP31( X )
% alpha     形状参数
% beta      尺度参数
% a0         位置参数
n = length(X);
meanX=mean(X);
K=X/meanX;
cv=sqrt(sum((K-1).^2)/(n-1));
cs=sum((K-1).^3) /((n-3)*cv^3);
alpha=4/cs^2;
beta=2/(cs*cv*meanX);
a0=meanX*(1-2*cv/cs);
disp(['EX,cv,cs',num2str([meanX,cv,cs])])
end


function [alpha , beta , a0,meanX,cv,cs] = fitP32( X )
% alpha     形状参数
% beta      尺度参数
% a0         位置参数
n = length(X);
meanX=mean(X);
K=X/meanX;
cv=sqrt(sum((K-1).^2)/(n-1));
sigma=std(X);

Ex=0;
Hx=0;
for i=1:n
    phi=normx( X(i),meanX,sigma);
    Ex=Ex+(X(i)-meanX)*phi;
    Hx=Hx+(X(i)-meanX)^2*phi;
end
cs=-4*sigma*Ex/Hx;

alpha=4/cs^2;
beta=2/(cs*cv*meanX);
a0=meanX*(1-2*cv/cs);
disp(['EX,cv,cs',num2str([meanX,cv,cs])])
end

function P = normx( x,xbar,sigma)
    P=exp(-(x-xbar)^2/(2*sigma^2))/(sqrt(pi*2)*sigma);
end


function P = P3cdf( X , alpha , beta , a0 )
% 利用P3分布得到相应分布函数

if isvector(X)
    n = length(X);
else
    error("输入数据并非向量！")
end

X=X(:);
Xmean = mean(X);                                         % 计算样本X的均值
K = X/Xmean;                                                % 计算模比系数K
Cv = sqrt((1/(n-1)).*sum((K-1).^2));                % 计算变差系数Cv
Cs = (sum((K-1).^3))/(Cv^3)/(n-3);                 % 计算偏态系数Cs
if Cs>0
    Y = X-a0;
else
    Y = a0-X;
end
Y = max(Y,0);          % Y<0则取值为0

if Cs>0
    P= gammainc(Y*beta,alpha,'lower');
else
    P= 1- gammainc(Y*beta,alpha,'lower');
end
end

function P = P3pdf( X , alpha , beta , a0)
n=length(X);
P =zeros(n,1);
for i=1:n
    P(i,1) = beta^alpha/gamma(alpha)*(X(i)-a0)*(alpha-1)*exp(-beta*(X(i)-a0));
end

end


function x = cdfcalc( px , EX,cv )
    x =[];
    for i=1:length(px)
        x=[x,EX*(1+cv*px(i))];
    end
end

function [a,b] = moment_norm(x)
    a=mean(x);
    b=length(x)/(length(x)-1)*std(x);

end

function all= cdfnorm(a,b)
all=[];
px=[0.01,1,10,20,30,50,60,80,90,99.9]/100;
for i=px
    try
        x=b(a<i);
        all=[all,x(end)];
    catch
        all=[all,0];
    end
    
end
end