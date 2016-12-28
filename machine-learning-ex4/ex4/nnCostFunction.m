function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(m, 1) X];                % add bias  - 5000 x 401
                                    % Theta1 - 25 x 401
z2 = a1 * Theta1';                  % 5000 x 25

a2 = [ones(m, 1) sigmoid(z2)];      % take sigmoid and add bias - 5000 x 26
                                    % Theta2 - 10 x 26
z3 = a2 * Theta2';                  % 5000 x 10
a3 = h = sigmoid(z3);               % 5000 x 10
k = num_labels;                     % output labels

% we need to recode the output labels as 1-hot encoding
y1 = zeros(m, k);                   % 5000 x 10
for i = 1:k
    y1(:, i) = y == i;              % set i_th column
end

J = 0;
% J = sum(sum(y1 .* log(h) + (1 - y1) .* log(1 - h)));
for i = 1:m
    for j = 1:k
        J += y1(i,j) * log(h(i,j)) + (1-y1(i,j)) * log(1 - h(i,j));
    end
end
J = -J/m;

% regularization excluding bias weights
penalty = (lambda/(2*m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));
J += penalty;

% -------------------------------------------------------------

error3 = h - y1;                    % 5000 x 10
error2 = error3 * Theta2(:, 2:end) .* sigmoidGradient(z2);    % 5000 x 25

delta1 = error2' * a1;
delta2 = error3' * a2;

Theta1_grad = delta1/m;
Theta2_grad = delta2/m;

Theta1_grad += (lambda/m) * [zeros(hidden_layer_size,1) Theta1(:, 2:end)];
Theta2_grad += (lambda/m) * [zeros(num_labels,1) Theta2(:, 2:end)];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
