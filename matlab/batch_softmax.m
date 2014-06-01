% X: size = row * col

function res = batch_softmax(X, row, col)

% rasterize X
X = reshape(X, row, col);
res = zeros(size(X));
for i = 1:col
    res(:, i) = softmax(X(:, i));
end

