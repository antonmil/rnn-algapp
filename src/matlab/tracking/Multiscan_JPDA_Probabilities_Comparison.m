function [Final_probabilty,diff,NVHypo,TNH, TimeComp]=Multiscan_JPDA_Probabilities_Comparison(M,Obj_info,mbest)

U=size(Obj_info,1);
Final_probabilty=cell(1,U);
diff=cell(max(mbest),1);
TimeComp=cell(2,1);

% Final_probabilty2=cell(1,U);
M2=sparse(M);
[~,C]=graphconncomp(M2,'Directed','false');
C2=C(1:U);
NR=cell2mat(cellfun(@(x) size(x.Prob,1),Obj_info,'UniformOutput', false));
NVHypo=zeros(1,length(unique(C2)));
TNH=zeros(1,length(unique(C2)));
for i=unique(C2)
    ix=(C2==i);
    NR_C=NR(ix);
    if size(NR_C,1)==1
        TNH(i)=NR_C;
    else
        TNH(i)=prod(NR_C);
    end
    %         Final_probabilty(ix)=MBest_JPDA_Probabilty_Calculator(Obj_info(ix),mbest);
    %         Final_probabilty2(ix)=JPDA_Probabilty_Calculator(Obj_info(ix));
    %         erro=cell2mat(cellfun(@(x,y) any(abs(x-y)>10^-5),Final_probabilty,Final_probabilty2,'UniformOutput', false));
    %         if any(erro)
    %         error('not equal')
    %         end

    [Final_probabilty(ix),NVHypo(i), alltime]=JPDA_Probabilty_CalculatorTime(Obj_info(ix),TNH(i));
    [Final_probabilty2, tends]=MBest_JPDA_Probabilty_CalculatorMulti(Obj_info(ix),min(max(mbest),NVHypo(i)));
        TimeComp{1,i}=alltime';
        TimeComp{2,i}=tends;
        for jj=1:min(max(mbest),NVHypo(i))
            if(NVHypo(i) > jj)
                errr=cell2mat(cellfun(@(x,y) abs(x-y),...
                    Final_probabilty(ix)',Final_probabilty2{jj}',...
                    'UniformOutput',false));
                diff{jj}=[diff{jj};[errr NVHypo(i)*ones(size(errr))]];
            end
        end
   if NVHypo(i)>10000
       disp(num2str(NVHypo(i)))
   end

    
end
