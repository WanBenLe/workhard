function w_j= shangquan(index,pot_ind)
    m=length(index);
    n=size(index,2);
    ptemp=index(:,1:pot_ind);
    parfor i =1:size(ptemp,2)
        temp1=ptemp(:,i);
        ptemp(:,i)=(temp1-min(temp1))/(max(temp1)-min(temp1))+0.02;
    end
    ntemp=index(:,(pot_ind+1):end);
    parfor i =1:size(ntemp,2)
        temp1=ntemp(:,i);
        ntemp(:,i)=(max(temp1)-temp1)/(max(temp1)-min(temp1))+0.02;
    end
    index=[ptemp,ntemp];
    
    f_ij=index./sum(index,1);
    k=1/log(m);
    H=-k.*sum(f_ij.*log(f_ij));
    w_j=(1-H)/(n-sum(H));
end

