function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha


% Initialize some useful values
m = length(y);       % number of training examples
% X = [ones(m, 1) X];  % add bias terms
J_history = zeros(num_iters, 1);
n = size(X, 2)       % number of features + bias

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    tmp = zeros(n, 1);
    for i = 1 : m
        for j = 1 : n
            for j = 1 : n
                h = theta(j,1)*X(i, j);
            end
            k = (h - y(i)) * X(i, n);
            tmp(i) = k;
        end
    end
    for i = 1 : n
        theta(i) = tmp(i) - alpha /m * k;
    end

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
