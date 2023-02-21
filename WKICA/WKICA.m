clear
clc
% 正常数据
% 制动踏板失效故障
% 电池短路故障
% 电机力矩传感器故障
% 电机短路故障

samplex=500;
zcdata=load('zhengchangData.mat').data';
zcdata=zcdata(1:samplex:length(zcdata),:);
n=length(zcdata);
disp(zcdata(1:3,1))
zcdata=zcdata(:,[2,4,6,10,11,13]);
zcdata=[ones(n,1),zcdata(:,2:end)];

zddata=load('zhidongData.mat').data';
zddata=zddata(1:samplex:length(zddata),:);
temp=ones(n,1);
temp(zddata(:,1)>=93)=2;
zddata=zddata(:,[2,4,6,10,11,13]);
zddata=[temp,zddata(:,2:end)];

dcdata=load('dianchiData.mat').data';
dcdata=dcdata(1:samplex:length(dcdata),:);
temp=ones(n,1);
temp(dcdata(:,1)>=94)=3;
dcdata=dcdata(:,[2,4,6,10,11,13]);
dcdata=[temp,dcdata(:,2:end)];


cgdata=load('chuanganData.mat').data';
cgdata=cgdata(1:samplex:length(cgdata),:);
temp=ones(n,1);
temp(cgdata(:,1)>=118)=4;
cgdata=cgdata(:,[2,4,6,10,11,13]);
cgdata=[temp,cgdata(:,2:end)];

zjdata=load('zajianData.mat').data';
zjdata=zjdata(1:samplex:length(zjdata),:);
temp=ones(n,1);
temp(zjdata(:,1)>=118)=5;
zjdata=zjdata(:,[2,4,6,10,11,13]);
zjdata=[temp,zjdata(:,2:end)];

all_data=[zddata;dcdata;cgdata;zjdata];
%打乱数据
N=length(all_data);
RandIndex=randperm(N);  
all_data=all_data(RandIndex,:); 
train_n=ceil(N*0.7);
train=all_data(1:train_n,:);
test=all_data(train_n+1:end,:);

%正常数据
zc_x=zscore(all_data(all_data(:,1)==1,2:end));
K=zc_x*zc_x';
N=size(zc_x,1);
K_=K - (2/N)*ones(N,N)*K + ((1/N)*ones(N,N))*K*((1/N)*ones(N,N));
K_white=white(K_,5);
Mdl3 = rica(K_white,5,'IterationLimit',400,'Standardize',true);
zc_x_new = transform(Mdl3,K_white);

%kde
[f,xi,bw] = ksdensity(zc_x_new(:,1),'Function','cdf', 'NumPoints',100);

figure;
plot(xi,f,'-g','LineWidth',2)
title('第1个独立分量KDE的CDF');
saveas(gcf, '第1个独立分量KDE的CDF', 'png');
a=[xi',f'];
a=a(a(:,2)>0.99);
a1=a(1,1);
%110.5958
[f,xi,bw] = ksdensity(zc_x_new(:,2),'Function','cdf', 'NumPoints',100);
figure;
plot(xi,f,'-g','LineWidth',2)
title('第2个独立分量KDE的CDF');
saveas(gcf, '第2个独立分量KDE的CDF', 'png');
a=[xi',f'];
a=a(a(:,2)>0.99);
a2=a(1,1);
%343.5421
[f,xi,bw] = ksdensity(zc_x_new(:,3),'Function','cdf', 'NumPoints',100);
figure;
plot(xi,f,'-g','LineWidth',2)
title('第3个独立分量KDE的CDF');
saveas(gcf, '第3个独立分量KDE的CDF', 'png');
a=[xi',f'];
a=a(a(:,2)>0.99);
a3=a(1,1);
%181.7732
[f,xi,bw] = ksdensity(zc_x_new(:,4),'Function','cdf', 'NumPoints',100);
figure;
plot(xi,f,'-g','LineWidth',2)
title('第4个独立分量KDE的CDF');
saveas(gcf, '第4个独立分量KDE的CDF', 'png');
a=[xi',f'];
a=a(a(:,2)>0.99);
a4=a(1,1);
%140.3099
[f,xi,bw] = ksdensity(zc_x_new(:,5),'Function','cdf', 'NumPoints',100);
figure;
plot(xi,f,'-g','LineWidth',2)
title('第5个独立分量KDE的CDF');
saveas(gcf, '第5个独立分量KDE的CDF', 'png');
a=[xi',f'];
a=a(a(:,2)>0.99);
a5=a(1,1);
%115.0437


%不太正常的数据
non_zc_x=zscore(all_data(all_data(:,1)~=1,2:end));
K=non_zc_x*non_zc_x';
N=size(non_zc_x,1);
K_=K - (2/N)*ones(N,N)*K + ((1/N)*ones(N,N))*K*((1/N)*ones(N,N));
K_white=white(K_,5);
non_zc_x_new = transform(Mdl3,K_white);
non_zc_x_neww=non_zc_x_new.*weight(non_zc_x_new,a1,a2,a3,a4,a5);
new_nonzc_data=[all_data(all_data(:,1)~=1,1),non_zc_x_neww];
new_zc_data=[all_data(all_data(:,1)==1,1),zc_x_new];
disp(1)

function w=weight(x,a1,a2,a3,a4,a5)
x1=x.^2;
RI2=exp(x1./mean(x1,2)-1);
w=ones(size(x,1),size(x,2));
w(x(:,1)>a1,1)=RI2(x(:,1)>a1,1);
w(x(:,2)>a2,2)=RI2(x(:,2)>a2,2);
w(x(:,3)>a3,3)=RI2(x(:,3)>a3,3);
w(x(:,4)>a4,4)=RI2(x(:,4)>a4,4);
w(x(:,5)>a5,5)=RI2(x(:,5)>a5,5);
end

function V= white(X,np)
X = X-ones(size(X,1),1)*mean(X);
covarianceMatrix = X*X'/size(X,2); 
[E, D] = eig(covarianceMatrix);
[~,order] = sort(diag(-D));
E = E(:,order);
d = diag(D); 
dsqrtinv = real(d.^(-0.5));
Dsqrtinv = diag(dsqrtinv(order));
%D = diag(d(order));
V = Dsqrtinv*E';
V=V(:,1:np);
end
