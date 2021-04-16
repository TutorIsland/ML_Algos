function [alpha, theta, J_prev, cont] = check_gradient_descent(J, J_prev, alpha, alpha_min, alpha_reducing_factor, theta)
    %CHECK_GRADIENT_DESCENT Check that cost J is decreasing
    %   Checks whether gradient descent is actually happening:
    %   if cost J is increasing, instead of decreasing,
    %   learning rate is reduced.
    %   If learning rate is already small enough,
    %   learning parameters are initialized and code continues to next iteration
    %   of the while loop - this function is "tied" to train_lin_reg.m .
	if (J > J_prev)
		if (alpha < alpha_min)
			[alpha, theta, J_prev] = init_learning_params();
            cont = true;
% 			continue;
            return;
		else
			alpha = alpha/alpha_reducing_factor;
		end
	else
% 		alpha = alpha;
	end
% 	theta = theta;
	J_prev = J;
end
