% if 'labels' is given, then we return a row matrix that 
% corresponds to the softmax probability at the correct label

function res = batch_softmax(X, labels)

[row, col] = size(X);

% calculate the full matrix
if ~exist('labels', 'var')
    res = zeros(size(X));
    for i = 1:col
        res(:, i) = softmax(X(:, i));
    end
% calculate only the probability at the correct label
else
    res = zeros(1, size(X, 2));
    for i = 1:col
        smx = softmax(X(:, i));
        res(i) = smx(labels(i) + 1);
    end
end

