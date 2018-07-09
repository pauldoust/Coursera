function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X  * theta;  % X: m, n+1  | theta n+1,1  = m, 1
thetareg= [0; theta(2:end)];
reg = (lambda / (2*m)) * (thetareg' * thetareg );
J = (1 / (2 * m) ) * ((h-y)' * (h - y) ) + reg; 


grad = (1/m) *  (X' * (h -y)    + (lambda * thetareg)); % m , 1 * m,n+1
%grad = (1/m) * (X' * (X * theta - y) + lambda * thetareg);





% =========================================================================

grad = grad(:);

end
