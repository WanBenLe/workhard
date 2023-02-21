function result = psoest(score1,score2)
para=[0.5 0.5 0.5 0.5 0.5];
r1=fun1(para,score1,score2);


r1iterx=[r1,para];
best_jb=r1iterx;
best_qj=r1iterx;
allp=[];
bond=3;
for quanju=1:10
    for jubu=1:40
        para=rand(1,5)*bond;

        para=check_para(para,bond);
        loss=fun1(para,score1,score2);
        r1iter=[loss,para];

        
        %更新参数
        r1iter(2:end)= r1iter(2:end)+ 2.8*rand()*(best_qj(2:end)-r1iter(2:end))+ 1.3*rand()*(best_jb(2:end)-r1iter(2:end));
        r1iter(2:end)=check_para(r1iter(2:end),bond);

        loss=fun1(r1iter(2:end),score1,score2);
        r1iter=[loss,r1iter(2:end)];

        if r1iter(1)<best_qj(1)
             best_qj=r1iter;
             best_jb=r1iter;
        elseif r1iter(1)<best_jb(1)
             best_jb=r1iter;
        end 
        if jubu==1
           if quanju==1
                best_qj=r1iter;
           end
           best_jb=r1iter;
        end

        loss=fun1(best_qj(2:end),score1,score2);
        r1iter=[loss,best_qj(2:end)];


        allp=[allp;best_qj(1)];
    end

end


result=best_qj;
%figure;
%plot(allp);
%title('PSO优化结果');
%saveas(gcf, 'PSO优化结果', 'png');
end




function p=check_para(p,bond)
    parfor i=1:5
        if (p(i)<0)||(p(i))>1
            p(i)=rand(1,1)*bond;
  
        end
    end
end
