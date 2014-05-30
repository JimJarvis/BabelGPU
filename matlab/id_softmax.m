% id(y == j) - softmax(alpha_vec)
%
function res = id_softmax(X, n)

X = X - max(X);
res = zeros(size(X));
res(n + 1) = 1;
X = exp(X);
res = res - X/sum(X);
