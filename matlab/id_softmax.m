% id(y == j) - softmax(alpha_vec)
%
function res = id_softmax(X, n)

m = max(X);
X = X - m;
res = zeros(size(X));
res(n) = 1;
X = exp(X);
res = res - X/sum(X);
