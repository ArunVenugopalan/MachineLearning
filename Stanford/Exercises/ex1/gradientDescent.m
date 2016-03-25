function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
k = length(theta);
% disp(k)
tmp_theta = theta;
for iter = 1:num_iters
  tmp = (X*theta) - y;
  for i = 1:k
   tmp_theta(i, 1)  = theta(i) - alpha*(1/m)* (sum(tmp.* X(:,i)));
  end
  % disp(tmp_theta)
     % tmp_theta(i) = theta(i) - alpha*(1/m)* (sum((X*theta) - y)) * X(2);
  theta = tmp_theta;

      
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end