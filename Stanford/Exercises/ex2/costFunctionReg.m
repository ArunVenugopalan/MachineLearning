function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% fprintf('Size of grad is %d %d ', (size(grad))); 

h_x = sigmoid(X*theta);

%disp(h_x);
term1 = (-y) .* (log(h_x));
term2 = (1-y) .* (log(1 - h_x));

thetaFrom1 = theta(2:end);
%disp(theta);

regularizedTerm = ((lambda)/(2*m)) * sum(thetaFrom1.^2);

%fprintf('Reguarized term size is %d %d ', (size(regularizedTerm))); 
%fprintf('Reguarized term value is %d %d ', regularizedTerm); 

J = ((1/m) * sum( term1 - term2)) + regularizedTerm;
errorTerm = (h_x - y); % m - column vector
% fprintf('Error term size is %d %d ', (size(errorTerm))); 
% disp(X);
% grad(1) = (1/m) * (errorTerm(1) * X(1,:)');

for i = 1:m
    grad = grad + (errorTerm(i) * X(i,:)');
end
grad = grad * (1/m);

%disp(grad)
%for i = 2:size(theta) - 1
    grad(2:end) = grad(2:end) + (lambda / m) .* theta(2:end);
%end
%disp(size(grad))

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
