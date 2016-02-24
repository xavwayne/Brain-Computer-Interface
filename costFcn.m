function [F, G, H] = costFcn(Z,X,Y,t,lamda)
% Compute the cost function F(Z)
%
% INPUTS:
%   Z: Parameter values
% OUTPUTS
%   F: Function value
%   G: Gradient value
%   H: Hessian value
%

[zr,~]=size(Z);
num=zr-205;%number of training examples
W=Z(1:204,1);
C=Z(205,1);
KS=Z(206:zr,1)';
item1=sum(KS)+lamda*(W'*W);
temp=W'*X.*Y+C*Y+KS-1;
item2=-sum(log(temp))/t;
item3=-sum(log(KS))/t;

%compute F
F=item1+item2+item3;

%compute G
% G for W
dif1_W=2*W*lamda;
dif2_W=zeros(204,1);
for i=1:num
    dif2_W=dif2_W+1/temp(i)*Y(i)*X(:,i);
end
dif2_W=-dif2_W/t;
G_W=dif1_W+dif2_W;

%G for C
G_C = -(1/t) * sum(Y./temp);
%G for KS
dif1_KS=ones(num,1);
dif2_KS=-(1./temp)'/t;
dif3_KS=-(1./KS)'/t;
G_KS=dif1_KS+dif2_KS+dif3_KS;

G=[G_W;G_C;G_KS];


%compute H
hs=zeros(zr,zr);
[r,~]=size(hs);
%first construct upper triangle
for i=1:r
    for j=i:r
        if(i<=204 && j==i)
            hs(i,j)=2*lamda+1/t*sum((Y./temp).^2.*X(i,:).*X(j,:));
        elseif (i<=204 && j<=204)
            hs(i,j)=1/t*sum((Y./temp).^2.*X(i,:).*X(j,:));
        elseif (i<=204 && j==205)
            hs(i,j)=1/t*sum((Y./temp).^2.*X(i,:));
        elseif (i<=204 && j>205)
            idx=j-205;
            hs(i,j)=1/t*Y(idx)*X(i,idx)/(temp(idx)^2);
        elseif (i==205 && j==i)
            hs(i,j)=1/t*sum((Y./temp).^2);
        elseif (i==205 && j>205)
            idx=j-205;
            hs(i,j)=1/t*Y(idx)/(temp(idx)^2);
        elseif (i>205 && j==i)
            idx=j-205;
            hs(i,j)=1/t/(temp(idx)^2)+1/t/(KS(idx)^2);
        else
            hs(i,j)=0;
        end
        
    end
end
%then construct lower triangle
hsT=hs';
for i=1:r
    hsT(i,i)=0;
end

H=hs+hsT;
end




