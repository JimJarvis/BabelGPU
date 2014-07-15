% X_: size = row * col
% labels: 0-based

function res = batch_id_softmax(X, labels)

[row, col] = size(X);
res = zeros(row, col);
for i = 1:col
    res(:, i) = id_softmax(X(:, i), labels(i));
end

