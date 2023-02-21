function [new_para,all_para] = EM(init_para,dy,dt,y,t)
diff=100;
all_para=[];
xx=0;
%参数变化和迭代次数上限
while (diff>10^-6)&&(xx<100000)
    xx=xx+1;
    %a.b.c.d
    a=init_para(1);
    b=init_para(2);
    c=init_para(3);
    d=init_para(4);
    %测量次数N
    N=size(dt,2);
    %产品数j
    j=size(dt,1);
    
    %根据产品数据估计期望,E步
    E_omega=zeros(j,1);
    E_ln_omega=zeros(j,1);
    E_omega_mu=zeros(j,1);
    E_omega_mu2=zeros(j,1);


    for i=1:j
        %N测量次数
        %j产品数量
        alpha=a+N/2;
        p1=b+c^2/(2*d);
        p2=0;
        p3_a=0;
        p3_b=0;
        for k=1:N
            p2=p2+dy(i,k)^2/(2*lambda(dt(i,k)));
            p3_a=p3_a+dy(i,k);
            p3_b=p3_b+lambda(dt(i,k));
        end
        p3=(c+d*sum(p3_a))^2/(2*(d+d^2* sum(p3_b)));
        beta=p1+p2-p3;
        E_omega(i)=alpha/beta;
        E_ln_omega(i)=psi(alpha)-log(beta);
        p4=(d*sum(p3_a)+c)/(sum(p3_b)*d+1);
        E_omega_mu(i)=E_omega(i)*p4;
        E_omega_mu2(i)= E_omega(i)*p4^2+p4;

    end
    
    %M步,就是单纯的更新系数
    fun = @(x) abs(psi(x)-log(x)-mean(E_ln_omega)-log(j)+log(sum(E_omega)));
    a1 = fminbnd(fun,0,100);
    if ~isnan(j*a1/sum(E_omega))
        b1=j*a1/sum(E_omega);
    else
        b1=b;
    end
    c1=sum(E_omega_mu)/sum(E_omega);
    d1=mean(E_omega_mu2-2*c1*E_omega_mu+c1^2*E_omega);
    
    new_para=[a1,b1,c1,d1];
    all_para=[all_para;new_para];
    %根据参数差决定是否继续迭代
    diff=max(abs(new_para-init_para));
    init_para=new_para;
end
end


function x=lambda(t)
x=t^1;
end
% 
% x=1
% psi(x)-log(x)-mean(E_ln_omega)+log(j)-log(sum(E_omega))
