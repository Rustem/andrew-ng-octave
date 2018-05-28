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
% add bias vector to training set
X_with_bias = [ones(size(X, 1), 1) X];
a1 = X_with_bias;
% Compute z2 and a2
a2 = sigmoid(a1 * Theta1');
a2 = [ones(size(a2, 1), 1) a2];
% Attach bias unit
% Compute z3 and a3
hx = a3 = sigmoid(a2 * Theta2');

% compute ymat by using 10x10 identity matrix: the output row is the value of vector y where y is in the interval [1, 10]:
%  y(i) = j   => [0, 0, 0, ... 1, ..., 0]
y_mat = eye(num_labels)(y, :);
% Compute cost function
J_if_y_1 = (-sum(sum(y_mat.*log(hx)))) / m;
J_if_y_0 = (-sum(sum((1 - y_mat).*log(1-hx)))) / m;
J = J_if_y_0 + J_if_y_1;

% regularizer will be added to the cost here
reg = (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2))) * (lambda / (2 * m));
J += reg;

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

[theta1_m, theta1_n] = size(Theta1);
[theta2_m, theta2_n] = size(Theta2);
D1 = zeros(theta1_m, theta1_n);
D2 = zeros(theta2_m, theta2_n);
for t = 1:m,
    % Part 1 forward pass
    a_1t = a1(t, :);
    a_2t = a2(t, :);
    a_3t = a3(t, :);
    % Part 2 compute output error
    d_3t = a_3t - y_mat(t, :);

    % Part 3 compute hidden layer errors
    d_2t = Theta2' * d_3t' .* [sigmoidGradient([1; Theta1 * a_1t'])];
    % d_2t = d_2t(2:end); % remove bias unit calculation
    % Part 4 accumulate the gradient
    D1 = D1 + d_2t(2:end) * a_1t;
    D2 = D2 + d_3t' * a_2t;
end;

Theta1_grad += (1/m) * D1 + (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_grad += (1/m) * D2 + (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
