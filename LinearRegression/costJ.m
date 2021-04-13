function [J] = costJ(diff_hy)
	m = length(diff_hy);
	J = ( sum(diff_hy) ).^2/(2.*m);
end
