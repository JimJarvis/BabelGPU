%% Generate test data for softmax step
% - softmax(alpha_vec)
% - softmax() only the probability at the correct label
% - Return the label where the maximum probability occurs in alpha_vec
%
TEST = 'Softmax';
loc = @(name) ['test/' TEST '_' name '.txt'];
dlm = @(val, str) dlmwrite(loc(str), val, 'precision', '%.16f');

%% DIMS
row = 50;
col = 300;
labelN = row-1; % if we use hasBias

X = unifrnd(-6, 6, row*col, 1);
Labels = randi(labelN, col, 1) - 1;

dlmwrite(loc('input_dim'), [row, col]);
dlm(X, 'input_X');
dlmwrite(loc('input_Labels'), Labels);

X = reshape(X, row, col);

% softmax() full
res = batch_softmax(X);
dlm(res, 'gold_batch_softmax');
% hasBias
res = batch_softmax(X(1:end-1, :));
dlm([res; zeros(1, col)], 'gold_batch_softmax_bias');

% softmax() - id
res = - batch_id_softmax(X, Labels);
dlm(res, 'gold_batch_softmax_minus_id');
res = - batch_id_softmax(X(1:end-1, :), Labels);
dlm([res; zeros(1, col)], 'gold_batch_softmax_minus_id_bias');

% softmax() for only the correct label and take log
res = log(batch_softmax(X, Labels));
dlm(res, 'gold_batch_softmax_at_label');
% sum of log likelihood
dlm(sum(res), 'gold_log_prob');

res = log(batch_softmax(X(1:end-1, :), Labels));
dlm(res, 'gold_batch_softmax_at_label_bias');
% sum of log likelihood
dlm(sum(res), 'gold_log_prob_bias');

% The best labels from X
[~, labelIdx] = max(X, [], 1);
labelIdx = labelIdx - 1;
dlmwrite(loc('gold_best_labels'), labelIdx);
[~, labelIdx] = max(X(1:end-1, :), [], 1);
labelIdx = labelIdx - 1;
dlmwrite(loc('gold_best_labels_bias'), labelIdx);

fprintf('Done!\n');
