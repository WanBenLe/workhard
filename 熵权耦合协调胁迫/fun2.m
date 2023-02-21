function yf = fun2(para,x)
    m=para(1);
    r=para(2);
    x1=m-r;
    b=para(3);
    a=para(4);
    p=para(5);
    %m,r,b,a,p
    yf=zeros(length(x),1);
    parfor i=1:length(x)
        x2=(x(i)-b)/a;
        yf(i)=x1*(10^x2-p)^2;
    end
end

