function logML = wienerML1(para,data)
%dt是1
mu=para(1)*lambda1(1);
sigma=(para(2)^2*lambda1(1))^0.5;
n=length(data);
logML=-(n/2)*log(2*pi)-n*log(sigma);
logML=logML-sum(((data-mu).^2)./(2*sigma^2));
%取负数是极大似然对数求极小方便
logML=-logML;
end

function x=lambda1(t)
x=t^1;
end
