function lossx = fun1(para,y,x)
    m=para(1);
    r=para(2);
    x1=m-r;
    b=para(3);
    a=para(4);
    p=para(5);
    %m,r,b,a,p
    loss=zeros(length(y),1);
    parfor i=1:length(y)
        x2=(x(i)-b)/a;
        loss(i)=(y(i)-x1*(10^x2-p)^2)^2;
    end
    lossx=sum(loss);
end

