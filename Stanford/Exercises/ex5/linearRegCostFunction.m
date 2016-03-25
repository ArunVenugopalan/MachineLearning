function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
h_x = X * theta; % prediction, m column vector
error = (h_x - y);
squaredError = error.^2;
thetaExcludingZero = [ 0; theta(2:end) ];
regularizedTerm = (lambda/(2*m)) * sum(thetaExcludingZero.^2);
J = (1/(2*m))* sum(squaredError) + regularizedTerm;
grad = (1/m)*(X' * error) + (lambda/m)*thetaExcludingZero;
grad = grad(:);

end
