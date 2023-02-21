clc
clear
%熵权法
data1=csvread('D:\项目\pythonProject6\ouhexietiao\data1.csv',1);
index1=data1(:,3:end);
shangquan1=shangquan(index1,9);
score1=index1*shangquan1';
data2=csvread('D:\项目\pythonProject6\ouhexietiao\data2.csv',1);
index2=data2(:,3:end);
shangquan2=shangquan(index2,2);
score2=index2*shangquan2';
%耦合协调
C=((score1.*score2)./(score1+score2).^2).^0.5;
T=0.5*score1+0.5*score2;
D=(C.*T).^0.5;
diff=score1-score2;
diff_type=diff;
%'g(y)滞后'
diff_type((diff>0.1))=1;
%'f(x)滞后'
diff_type(diff<-0.1)=2;
%'均衡'
diff_type(abs(diff)<0.1)=3;
D_type=D;
%'优质协调';'中级协调';
D_type(D>0.8)=1;
D_type((0.7<D)&(D<=0.8))=2;
D_type((0.6<D)&(D<=0.7))=3;
D_type((0.5<D)&(D<=0.6))=4;
D_type((0.4<D)&(D<=0.5))=5;
D_type((0.3<D)&(D<=0.4))=6;
D_type((0.2<D)&(D<=0.3))=7;
D_type(D<=0.2)=8;
%城市编号,年份,分数1,分数2
type_result=[data1(:,1),data1(:,2),score1,score2,D,D_type,diff,diff_type];
%[data1(:,1),data1(:,2),score1,score2];可以出图1a.b
%每座城市算score1,score2增长率求平均可以出cd
%[data1(:,1),data1(:,2),D_type,diff_type]其实是表4


%交互胁迫模型
city_unique=unique(data1(:,1));
para_result=zeros(length(city_unique),6);
para_result(:,1)=city_unique;
for i=1:length(city_unique)
    temp=type_result(type_result(:,1)==i,3:4);
    score1_t=temp(:,1);
    score2_t=temp(:,2);
    result=psoest(score1_t,score2_t);
    %     m=para(1);
    %     r=para(2);
    %     b=para(3);
    %     a=para(4);
    %     p=para(5);
    para_result(i,2:end)=result(2:end);
    yf=fun2(result(2:end),score2_t);
    close all
    figure
    plt=sort([score2_t,yf]);
    plot(plt(:,1),plt(:,2));
    hold on ;
    plt=sort([score2_t,score1_t]);
    scatter(plt(:,1),plt(:,2),'r')
    title('拟合与真实值');
    saveas(gcf,  strcat('拟合与真实值',num2str(i)), 'png');

end
%para_result可以出每座城市的参数估计结果,图2表3


%
data3=csvread('D:\项目\pythonProject6\ouhexietiao\data3.csv',1);
d_mat=data3(:,2:end);
year_unique=unique(data1(:,2));

woow=zeros(length(city_unique)*length(year_unique),4);
ix=1;
for i=1:length(city_unique)
    for j=1:length(city_unique)
         temp1=type_result(type_result(:,1)==i,5);
         temp2=type_result(type_result(:,1)==j,5);
         R_ij=1*temp1.*temp2/d_mat(i,j)^2;
         woow(ix:ix+length(R_ij)-1,4)=R_ij;
         woow(ix:ix+length(R_ij)-1,3)=year_unique;
         woow(ix:ix+length(R_ij)-1,1)=i;
         woow(ix:ix+length(R_ij)-1,2)=j;
         ix=ix+length(R_ij);

    end
end
woow=woow(woow(:,4)~=inf,:);
%每年每个城市的Ri(参考公式8),结果是图3
for i=1:length(year_unique)
    disp(year_unique(i))
    for j=1:length(city_unique)
        temp=woow((woow(:,1)==city_unique(j))&(woow(:,3)==year_unique(i)),4);
        disp(sum(temp))
    end
end
disp(1)

