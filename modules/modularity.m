A = readmatrix("matrix_0.csv");
[Ci,Q] = modularity_und(1-A,1);
writematrix(Ci,'matrix_0_modules.csv')