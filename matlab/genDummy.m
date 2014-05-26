%% Write a random matrix within range to CSV
%
loc = @(name) [name '.txt'];

%% DIMs
samples = 3000; % total number of training
x_dim = 36;
x_new_dim = 100;
labels = 50;

%% Learning constants
learning_rate = 1.5;
lambda = 2;

% X needs to be transposed to add a column of 1
X = unifrnd(-6, 6, samples, x_dim);

% W and b are put together
% X needs to be augmented with an extra 1
W = unifrnd(-3, 3, x_new_dim, x_dim+1);

% Theta will be labels * x_new_dim
Y = randi(labels, samples, 1) - 1;

dlm = @(val, str) dlmwrite(loc(str), val, 'precision', '%.15f');

dlm(X, 'input_X');
dlm(W, 'input_W');

dlmwrite(loc('input_Y'), Y);
dlmwrite(loc('input_dim'), [samples, x_dim, x_new_dim, labels]);
dlmwrite(loc('input_learn'), [learning_rate, lambda]);


%%%% Testing with the training process
% Augment with 1
X1 = [X ones(samples, 1)];

%% Step 1
Xnew = cos(W * X1');

dlm(Xnew, 'gold_Xnew');

%% Step 2
Theta = zeros(labels, x_new_dim);

for s = 1:samples
    Xnew_s = Xnew(:, s);
    A = Theta * Xnew_s;
    A = id_softmax(A, Y(s));

    %% Step 3, update Theta
    Theta = Theta + learning_rate * (A * Xnew_s' - lambda/samples * Theta);
end
dlm(A, 'gold_A');

fprintf('Done!\n');
dlm(Theta, 'gold_Theta');
