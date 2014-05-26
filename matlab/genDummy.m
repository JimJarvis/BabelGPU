%% Write a random matrix within range to CSV
%
loc = @(name) [name '.txt'];

samples = 3000; % total number of training
x_dim = 36;
x_new_dim = 100;
labels = 50;

% X needs to be transposed to add a column of 1
X = unifrnd(-6, 6, samples, x_dim);

% W and b are put together
% X needs to be augmented with an extra 1
W = unifrnd(-3, 3, 100, x_dim+1);

% Theta will be labels * x_new_dim
Y = randi(labels, samples, 1) - 1;

dlmwrite(loc('test_X'), X);
dlmwrite(loc('test_W'), W);
dlmwrite(loc('test_Y'), Y);
dlmwrite(loc('test_dim'), [samples, x_dim, x_new_dim, labels]);
