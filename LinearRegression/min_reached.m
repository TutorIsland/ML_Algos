% Restituisce vero solo se entrambe le componenti del gradiente
% di J sono < tol; altrimenti restituisce falso.
function [yn] = min_reached(gradJ_, tol)
	yn = (abs(gradJ_(1)) < tol) && (abs(gradJ_(2)) < tol);
end
