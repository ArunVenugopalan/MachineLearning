function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values

m = length(y); % number of training examples
h_x = sigmoid (X * theta);

temp = theta;
temp(1) = 0;

regularizedTerm = ((lambda)/(2*m)) * sum(temp.^2);

term1 = (-y) .* (log(h_x));
term2 = (1-y) .* (log(1 - h_x));
J = (1/m) * sum( term1 - term2) + regularizedTerm;
disp(J);
errorTerm = (h_x - y); % m - column vector

grad = (1/m) * X' * errorTerm;
grad = grad + (lambda/m) * temp;

grad = grad(:);

end
