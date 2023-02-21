function logML = wienerML2(para,data)
%dt是1
mu=para(1)*lambda2(1,para(3));
sigma=(para(2)^2*lambda2(1,para(3)))^0.5;
n=length(data);
logML=-(n/2)*log(2*pi)-n*log(sigma);
logML=logML-sum(((data-mu).^2)./(2*sigma^2));
%取负数是极大似然对数求极小方便
logML=-logML;

end

function x=lambda2(t,c)
x=t^c;
end
