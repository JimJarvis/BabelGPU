%% Generate test data for mini-batch (id - softmax) step
%
loc = @(name) [name '.txt'];
dlm = @(val, str) dlmwrite(loc(str), val, 'precision', '%.16f');

%% DIMS
row = 1000;
col = 250;
labels = 50;

X_ = unifrnd(-6, 6, row*col, 1);
Y = randi(labels, row*col, 1) - 1;

res = batch_id_softmax(X_, row, col, Y);

dlmwrite(loc('input_dim'), [row, col, labels]);
dlm(X_, 'input_X');
dlmwrite(loc('input_Y'), Y);

dlm(res, 'gold_MB');
fprintf('Done!\n');
