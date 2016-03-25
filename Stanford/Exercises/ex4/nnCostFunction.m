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

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

a1 = [ones(m, 1) X]; % 5000 x 401 matrix
z2 = Theta1 * a1'; % 25 x 5000 matrix
a2 = sigmoid(z2); % 25 x 5000 matrix
oneColumnVec = (ones(size(a2, 2), 1));

a2 = [oneColumnVec' ; a2 ]; % 26 x 5000 matrix 
z3 = Theta2 * a2; % 10 x 5000 matrix
a3 = sigmoid(z3); % 10 x 5000 matrix
h_x = a3'; % 5000 x 10 matrix

Y=zeros(size(h_x));
for i=1:m
    Y(i,y(i))=1;   
end
term1 = -Y .* (log(h_x)); % 5000 x 10 matrix
term2 = (1-Y) .* (log(1 - h_x)); % 5000 x 10 matrix
sumOfKClasses = sum((term1 - term2), 2); % 5000 x 1 matrix

J = (1/m) * sum( sumOfKClasses);

temp1 = Theta1;
temp1(:,1)= zeros(size(temp1, 1), 1);

temp2 = Theta2;
temp2(:,1)= zeros(size(temp2, 1), 1);

regularizedTerm = (lambda/(2*m)) * ((sum(sum(temp1 .* temp1)) + sum(sum(temp2 .* temp2))));
J = J + regularizedTerm;

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
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% delta3 = h_x - Y; % 5000 x 10
% delta2 = delta3 * Theta2; % 5000 x 26
% delta2(:,1) = zeros(size(delta2, 1), 1); % 5000 x 25
% delta2 = delta2 .* sigmoidGradient(z2); % 

z_2 = z2'; % convert into 5000 x 10
a_1 = a1; % 5000 x 400
a_2 = a2;
%a_2(1,:) = []; %zeros(size(a_2, 1), 1);
a_2 = a_2'; % 5000 x 25

Theta1_grad_temp = zeros(size(Theta1));
Theta2_grad_temp = zeros(size(Theta2));

for t=1:m % for each training example
    %disp(size(h_x(t)));
    z_2t = z_2(t,:)';
    delta3 = (h_x(t,:) - Y(t,:))'; % 10 col vec
    %disp(size(delta3));
    %disp(size(Theta2'));
    
    delta2 = (Theta2' * delta3); % 26 col vec
    delta2 = delta2(2:end);
    delta2 = delta2 .* sigmoidGradient(z_2t);  % 25 row vec
    %disp(size(z_2t));
    %disp(size(delta3));
    %disp(size(delta3 * a_2(t)));
     
%     disp(size((delta2 * a_1(t,:))));
%     disp(size(Theta1_grad_temp));
    
%     disp(size(a_2));
%     disp(size((delta3 * a_2(t,:))));
%     disp(size(Theta2_grad_temp));

    Theta1_grad_temp = Theta1_grad_temp + (delta2 * a_1(t,:)); % 25 x 400 matrix
    Theta2_grad_temp = Theta2_grad_temp + (delta3 * a_2(t,:)); % 10 x 25 matrix
end

modifiedTheta1 = Theta1;
modifiedTheta1(:,1) = zeros(size(Theta1, 1), 1); 

modifiedTheta2 = Theta2;
modifiedTheta2(:,1) = zeros(size(Theta2, 1), 1); 


Theta1_grad = (1/m) * Theta1_grad_temp + (lambda/m) * modifiedTheta1;
Theta2_grad = (1/m) * Theta2_grad_temp + (lambda/m) * modifiedTheta2;

%Theta1_grad(:,1) = zeros(size(Theta1_grad, 1), 1);
%Theta2_grad(:,1) = zeros(size(Theta2_grad, 1), 1);
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
