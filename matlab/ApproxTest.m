% Kernel approximation test
%
kernel = 'gauss';
newDim = 200000;
oldDim = 360;
N = 1000;
lambda = 0.001; % gamma parameter
verbose = false;
Xmean = 3;

X = rand(oldDim, N) * Xmean - Xmean/2;
W = kernelApprox(oldDim, newDim, lambda, kernel);

% normalize w.r.t. #samples
X_new = cos(W * [X; ones(1, N)]) * sqrt(2/newDim);

sumPerError = 0;
for i = 1:N-1
    x1 = X(:, i);
    x2 = X(:, i+1);
    exact = kernelExact(x1, x2, lambda, kernel);
    approx = dot(X_new(:, i), X_new(:, i+1));
    dif = abs(exact - approx);
    if verbose
        fprintf('%.3f   %.3f   %.3e\n', exact, approx, dif);
    end
    sumPerError = sumPerError + dif/exact;
end

fprintf('Percentage error: %.3f%%\n', sumPerError/(N - 1) * 100);
