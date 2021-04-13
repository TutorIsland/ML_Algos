function [gJ] = gradJ(diff_hy, x)
	gJ = zeros(2,1);
	m = length(diff_hy);
	gJ(1) = sum(diff_hy)/m;
	gJ(2) = (diff_hy'*x)/m;
end
