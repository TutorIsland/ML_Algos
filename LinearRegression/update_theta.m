function [theta_] = update_theta(theta, alpha, gradJ_)
		theta_ = zeros(2, 1);
		theta_(1) = theta(1) - alpha*gradJ_(1);
		theta_(2) = theta(2) - alpha*gradJ_(2);
end
