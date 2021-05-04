for no=1:10
    A = readmatrix(sprintf("networks/matrix_%d.csv",no));
    [Ci,Q] = modularity_und(1-A,1);
    writematrix(Ci,sprintf("networks/matrix_%d_modules.csv",no))
end