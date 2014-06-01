% softmax(alpha_vec)
%
function X = softmax(X, n)

X = X - max(X);
X = exp(X);
X = X / sum(X);
