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

a1 = [ones(m,1) X]; % 5000 , 401

z2 = a1 *  Theta1';  %a1 & theta 1 =  5000,401 * 25,401'  = 501, 25
a2 = sigmoid(z2); %5000, 25

a2 = [ones(m,1) a2];
z3 = a2 * Theta2'; %5000, 26 * 10, 26' = 5000 * 10
a3 = sigmoid(z3);
h = a3;

Y = zeros(m,num_labels);
for i = 1:m
    Y(i,y(i)) = 1;
	cost(i) = -Y(i,:) * log(h(i,:))' - ( (1-Y(i,:))  * log(1 - h(i,:))')  ;
end
%size(-Y(i,:) * log(h(i,:))')
% 5000,10 * 5000,10
%cost2 = -Y * log(h)' - ( (1-Y)  * log(1 - h)');
%cost2 = -Y * log(h)';
%size(Y)
%size(h)
%size(cost)
%size(cost2)
J = 1/m * sum(cost);
% (5000,1)' * (5000,10) - (5000,1)' * (5000,10) 
% (5000,10)' * (5000,10) - (5000,10)' * (5000,10) 
%J = ((-y)'*log(h)- (1-y)'*log(1-h) )  /m  ;
%J  = - y * log(H); % 5000, 10 * (5000 10)' =  5000


layer_1 = Theta1(:,2:end) .^ 2;
layer_2 = Theta2(:,2:end) .^ 2;

sum_layer_1 = sum(layer_1(:));
sum_layer_2 = sum(layer_2(:));

J= J + (lambda/(2*m)) * (sum_layer_1 + sum_layer_2);
% -------------------------------------------------------------

% =========================================================================
s3 = a3 - Y;
size(s3* Theta2)
size(sigmoidGradient([ones(m,1) z2]))
s2 = (s3 * Theta2 ) .* sigmoidGradient([ones(m,1) z2]); %10,26 *  5000,10 | 5000 10 * 10 26= 5000 26 .* 5000 26
s2 = s2(:,2:end);
delta_1 = (s2' * a1);
delta_2 = (s3' * a2);

p1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
p2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
Theta1_grad = delta_1 ./ m + p1;
Theta2_grad = delta_2 ./ m + p2;
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
