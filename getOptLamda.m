function optLamda = getOptLamda(X, Y, setPara)
% Get the optimal lamda
%
% INPUTS:
%   X(MxN) : trData(i,j) is the i-th feature from the j-th trial
%   Y(1xN): trData(j) is the label of the j-th trial (1 or -1)
%   setPara : Initialized parameters
%            setPara.t
%            setPara.beta
%            setPara.Tmax
%            setPara.tol
%            setPara.W
%            setPara.C
%
% OUTPUTS:
%   optiLamda: Optimal lamda value


lamda_pool=[0.01,1,100,10000];
acc=zeros(1,4);
%for every lammda, do five-fold cross validation
for k=1:4
    lamda=lamda_pool(k);
    Ac=zeros(1,5);
    for i=1:5
        % init_Z = [W, C, zeta];
        W=setPara.W;
        C=setPara.C;
        t=setPara.t;
        tol=setPara.tol;
        beta=setPara.beta;
        Tmax=setPara.Tmax;
        KS=zeros(1,240*5/6*4/5);
        %devide training data and test data
        idx=(i-1)*20+1:(i-1)*20+20;
        test_idx=[idx,idx+100];
        all=(1:240*5/6);
        train_idx=setdiff(all,test_idx);
        X_train=X(:,train_idx);
        Y_train=Y(:,train_idx);
        X_test=X(:,test_idx);
        Y_test=Y(:,test_idx);
        
        %IPM
        %initial guess
        for j=1:240*5/6*4/5
            tmp=1-(W'*X_train(:,j)+C)*Y_train(j);
            KS(j)=max(tmp,0)+0.001;
        end
        Z=[W;C;KS'];
        
        while(t<Tmax)
            %solve unconstrained nonlinear optimization
            [opt,~]=solveOptProb_NM(@costFcn,Z,tol,X_train,Y_train,t,lamda);
            Z=opt;
            t=beta*t;
        end
        W_opt=Z(1:204,1);
        C_opt=Z(205,1);
        %predict and calculate accuracy
        value=W_opt'*X_test+C_opt;
        [~,cols]=size(value);
        predict=ones(1,cols);
        predict(value<0)=-1;
        diff=predict-Y_test;
        correct=sum(diff==0);
        Ac(i)=correct/cols;
    end
    Ave_Ac=mean(Ac);
    acc(k)=Ave_Ac;
    
end
[~,col]=max(acc);
optLamda=lamda_pool(col);

