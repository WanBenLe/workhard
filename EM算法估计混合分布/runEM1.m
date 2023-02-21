clc
clear

data=xlsread('train2.xls');
%注释的代码是原数据的
%data=xlsread('train2.xls');
%data(:,3)=(data(:,3)-min(data(:,3)))/(max(data(:,3))-min(data(:,3)));
tloop=reshape(data(:,2),16,10)';
yloop=reshape(data(:,3),16,10)';
%处理成行是产品,列是测量次数
%tloop=reshape(data(:,2),100,100)';
%yloop=reshape(data(:,3)',100,100)';
affxxx=[];
affmse=[];
allpara=[];
%预测数据和归一
data1=xlsread('test1.xls');
t1=data1(:,2);
y1=data1(:,3);
%y1=(y1-min(y1))/(max(y1)-min(y1));
%ytrue=fliplr(data1(9:10:78,4));
ytrue=data1(:,4);
%AIC
t=tloop;
y=yloop;
%归一
% for i=1:size(y,1)
%        y(i,:)=(y(i,:)-min(y(i,:)))/(max(y(i,:))-min(y(i,:)));
% end

dt=t(:,2:end)-t(:,1:(end-1));

dy=y(:,2:end)-y(:,1:(end-1));
% dt=[zero;(size(t,1),1),dt];
% dy=[zeros(size(y,1),1),dy];
y=y(:,2:end);
t=t(:,2:end);
%wiener过程估计,2个跟正态分布相关的参数一个带阶数,见wienerML1.wienerML2
% 最小的就是AIC1就是lambda=t,图可以自行保存
wiener_fit_data=dy(1:end)';
fun = @(x) wienerML1(x,wiener_fit_data);
options = optimset('PlotFcns',@optimplotfval);
[para1,NlogML1] = fminsearch(fun,[0,0.27],options);
AIC1=2*NlogML1+2*2;
saveas(gcf, 'ML1', 'png');
fun = @(x) wienerML2(x,wiener_fit_data);
options = optimset('PlotFcns',@optimplotfval);
[para2,NlogML2] = fminsearch(fun,[1,1,3],options);
AIC2=2*NlogML2+2*3;
saveas(gcf, 'ML2', 'png');
%获得置信区间的有放回抽样,可以取大一些
paraall=[];
loopx=30;
for loop=1:loopx
    arand=ceil(size(yloop,1)*rand(1,size(yloop,1)));
    t=tloop(arand,:);
    y=yloop(arand,:);
    %t=tloop;
    %y=yloop;
    %归一
%      for i=1:size(yloop,1)
%           y(i,:)=(y(i,:)-min(y(i,:)))/(max(y(i,:))-min(y(i,:)));
%      end
    
    dt=t(:,2:end)-t(:,1:(end-1));
    
    dy=y(:,2:end)-y(:,1:(end-1));
    % dt=[zero;(size(t,1),1),dt];
    % dy=[zeros(size(y,1),1),dy];
    y=y(:,2:end);
    t=t(:,2:end);
    %给定初始参数进行估计
    init_para=[1,2,3,4];
    %EM估计看EM
    [new_para,all_parax]=EM(init_para,dy,dt,y,t);
    %new_para=[0.05,0.06,0.7,0.8]
    %new_para=[5.121,4.144*10^-4,2.025*10^-3,2.217*10^-3]
    allf=[];
    %取7个点进行预测并与真实值比较
    for xxx=1:length(data1)
        dt1=t1(2:xxx)-t1(1:(xxx-1));
        dt1= [0;dt1];
        dy1=y1(2:xxx)-y1(1:(xxx-1));
        dy1= [0;dy1];
        t1a=t1(1:xxx);
        y1a=y1(1:xxx);
%         dt1=dt1(end);
%         dy1=dy1(end);
%         t1a=t1a(end);
%         y1a=y1a(end);
        %预测看EMupf
        [forcast,new_parax]=EMupf(new_para,y1a,t1a,dy1,dt1);
        paraall=[paraall;new_parax];
        allf=[allf;forcast];
    end
    allf=(allf);
    affxxx=[affxxx,allf];
    allpara=[allpara;new_para];
end
%kde出概率密度,
[f1,xi1] =ksdensity(affxxx(1,:));
[f2,xi2] =ksdensity(affxxx(2,:));
[f3,xi3] =ksdensity(affxxx(3,:));
[f4,xi4] =ksdensity(affxxx(4,:));
[f5,xi5] =ksdensity(affxxx(5,:));
disp(mean(affxxx'))
disp(ytrue')
%disp(mean(abs(mean(affxxx')/ytrue'-1)))

%paraset是系数结果集抄一下就好
%行:测量次数
%列:4个参数的5%,估计系数均值,95%
paraall1=reshape(paraall',4,length(data1),loopx);
paraset=zeros(length(data1),4);
for i=1:length(data1)
    for j=1:4
    temp=reshape(paraall1(j,i,:),loopx,1);
    paraset(i,1+(j-1)*3)=prctile(temp,5);
    paraset(i,2+(j-1)*3)=mean(temp);
    paraset(i,3+(j-1)*3)=prctile(temp,95);
    end
end

%百分位数出可靠度曲线
probx=zeros(length(data1),100);
parfor i=1:length(data1)
    temp=affxxx(i,:);
    for j=1:100
        probx(i,j)=prctile(temp,j);
    end    
end
y1=100:-1:1;
plot3(probx(1,:)',ones(length(xi1))*500,y1, ...
    probx(2,:)',ones(length(xi1))*1000,y1,...
    probx(3,:)',ones(length(xi1))*1500,y1, ...
    probx(4,:)',ones(length(xi1))*2000,y1, ...
    probx(5,:)',ones(length(xi1))*2500,y1)
title('可靠度曲线');
xlabel('预测寿命');
ylabel('测量时间');
set(gca,'XGrid', 'on', 'YGrid', 'on', 'ZGrid', 'on');
try
    saveas(gcf, '可靠度曲线', 'png');
catch
    disp(1)
end

plot3(xi1',ones(length(xi1))*500,f1', ...
    xi2',ones(length(xi1))*1000,f2',...
    xi3',ones(length(xi1))*1500,f3', ...
    xi4',ones(length(xi1))*2000,f4', ...
    xi5',ones(length(xi1))*2500,f5')
hTitle = title('概率密度曲线');
hXLabel = xlabel('预测剩余寿命');
hYLabel = ylabel('测量时间');
set(gca,'XGrid', 'on', 'YGrid', 'on', 'ZGrid', 'on');
saveas(gcf, '概率密度曲线', 'png');
clf
plot(all_parax(:,1))
saveas(gcf, '参数1迭代', 'png');
plot(all_parax(:,2))
saveas(gcf, '参数2迭代', 'png');
plot(all_parax(:,3))
saveas(gcf, '参数3迭代', 'png');
plot(all_parax(:,4))
saveas(gcf, '参数4迭代', 'png');
disp(1)
