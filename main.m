% THIS IS THE MAIN FUNCTION TO RUN THE PROJECT.
tic

load('feaSubEOvert');
%load('feaSubEImg');

%initiallization
%load 120*2 trials of 204 features
X=[class{1},class{2}];
%label the data
Y=[ones(1,120),-ones(1,120)];
Ac=zeros(1,6);
optLamda=zeros(1,6);
%six-fold cross validation
for i=1:6
    W=ones(204,1);
    C=0;
    t=1;
    tol=0.000001;
    beta=15;
    Tmax=1000000;
    KS=zeros(1,240*5/6);
    
    setPara.tol=tol;
    setPara.beta=beta;
    setPara.Tmax=Tmax;
    setPara.t=t;
    setPara.W=W;
    setPara.C=C;
    %devide training data and test data
    idx=(i-1)*20+1:(i-1)*20+20;
    test_idx=[idx,idx+120];
    all=(1:240);
    train_idx=setdiff(all,test_idx);
    X_train=X(:,train_idx);
    Y_train=Y(:,train_idx);
    X_test=X(:,test_idx);
    Y_test=Y(:,test_idx);
    %get optimal lamda
    optLamda(i)=getOptLamda(X_train,Y_train,setPara);
    %IPM
    %initial guess
    for j=1:240*5/6
        tmp=1-(W'*X_train(:,j)+C)*Y_train(j);
        KS(j)=max(tmp,0)+0.001;
    end
    Z=[W;C;KS'];
    
    while(t<Tmax)        
        %solve unconstrained nonlinear optimization
        [opt,err]=solveOptProb_NM(@costFcn,Z,tol,X_train,Y_train,t,optLamda(i));        
        Z=opt;
        t=beta*t;
    end
    W_opt=Z(1:204,1);
    if(i==1)
        Z_sava=Z;
        show_chanWeights(abs(W_opt));
    end
    C_opt=Z(205,1);
    %predict and calculate error
    value=W_opt'*X_test+C_opt;
    [~,cols]=size(value);
    predict=ones(1,cols);
    predict(value<0)=-1;
    diff=predict-Y_test;
    correct=sum(diff==0);
    Ac(i)=correct/cols;
    
end

Ave_Ac=mean(Ac);
stdAc=std(Ac);
toc
