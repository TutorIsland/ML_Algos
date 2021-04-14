% Calcola il valore delle prediction,
% dati i valori di theta (intercetta e coeff. ang.)
% e degli examples x.
%
% La function si aspetta che x sia un vettore colonna di m elementi
% e theta sia un vettore colonna di 2 elementi [theta0 theta1]'.
%
function [h] = predict(theta, x)
	m = length(x);
	% aggiungo una colonna di m "uni" al vettore x
	xx = [ones(m,1) x];

	h = xx*theta;
end
