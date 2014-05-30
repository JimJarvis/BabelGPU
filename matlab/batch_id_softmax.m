% X_: size = row * col
% labels: 0-based

function res = batch_id_softmax(X_, row, col, labels)

% rasterize X
X = reshape(X_, row, col);
res = zeros(size(X));
for i = 1:col
    res(:, i) = id_softmax(X(:, i), labels(i));
end

