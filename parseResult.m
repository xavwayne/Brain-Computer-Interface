close all;
w_opt=Z_sava(1:204,1);
c_opt=Z_sava(205,1);
bar(w_opt)
figure
show_chanWeights(abs(w_opt))