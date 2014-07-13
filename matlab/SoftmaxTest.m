%% Generate test data for softmax step
% - softmax(alpha_vec)
% - softmax() only the probability at the correct label
% - Return the label where the maximum probability occurs in alpha_vec
%
TEST = 'Softmax';
loc = @(name) ['test/' TEST '_' name '.txt'];
dlm = @(val, str) dlmwrite(loc(str), val, 'precision', '%.16f');

%% DIMS
row = 1000;
col = 250;
labels = row;

X = unifrnd(-6, 6, row*col, 1);
Y = randi(labels, col, 1) - 1;

dlmwrite(loc('input_dim'), [row, col]);
dlm(X, 'input_X');
dlmwrite(loc('input_Y'), Y);

% softmax() full
res = batch_softmax(X, row, col);
dlm(res, 'gold_softmax');

% softmax() for only the correct label
res = batch_softmax(X, row, col, Y);
dlm(res, 'gold_softmax_labeled');

% sum of log likelihood
dlm(sum(log(res)), 'gold_log_prob');

% The best labels from X
[~, labelIdx] = max(reshape(X, row, col), [], 1);
labelIdx = labelIdx - 1;
dlmwrite(loc('gold_best_labels'), labelIdx);

fprintf('Done!\n');
