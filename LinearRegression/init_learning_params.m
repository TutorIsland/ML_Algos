% Initializes the value for
% alpha: learning rate used during Gradient Descent
% theta: (2x1) array containing guess for theta0 and theta1
% J_prev: cost of previous iteration set to max value possible
function [alpha theta J_prev gradJ_ max_iter alpha_min alpha_reducing_factor] = init_learning_params(tol)
	alpha = 15.;
	theta = rand(2,1);
	% setta J_prev al valore piu' grande rappresentabile
	J_prev = realmax;
	gradJ_ = [2.*tol 2.*tol]';
	% these params can eventually be exposed in the API
	max_iter = 10000;
	alpha_min = 1e-8;
	alpha_reducing_factor = 2.;
end
