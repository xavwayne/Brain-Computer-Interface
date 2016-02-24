function [optSolution, err] = solveOptProb_NM(costFcn,init_Z,tol,X,Y,t,lamda)
% Compute the optimal solution using Newton method
%
% INPUTS:
%   costFcn: Function handle of F(Z)
%   init_Z: Initial value of Z
%   tol: Tolerance
%
% OUTPUTS:
%   optSolution: Optimal soultion
%   err: Errorr

Z = init_Z;
err = 1;
[zr,~]=size(Z);
num=zr-205;%number of training examples

% Set the error 2*tol to make sure the loop runs at least once
while (err/2) > tol
    
    
    % Execute the cost function at the current iteration
    % F : function value, G : gradient, H, hessian
    [~, G, H] = feval(costFcn,Z,X,Y,t,lamda);
    
    [r,~]=size(H);
    E=eye(r);
    H_inv=H\E;
    %compute newton step and decrement
    dZ=-H_inv*G;
    err=-G'*dZ;
    
    %line search
    s=1;
    while(1==1)
        tmp=Z+s*dZ;
        W=tmp(1:204,1);
        C=tmp(205,1);
        KS=tmp(206:zr,1)';
        
        flag=1;
        for i=1:num
            a=W'*X(:,i)*Y(i)+C*Y(i)+KS(i)-1;
            b=KS(i);
            if (a<=0 || b<=0)
                flag=0;
                break;
            end
        end
        if (flag==1)
            break;
        end
        s=s*0.5;
    end
    %update
    Z=Z+s*dZ;
    
end
optSolution=Z;
end



