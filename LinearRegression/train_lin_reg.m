% A partire dal dataset in input (feature x, outcome noti y),
% usa una tolleranza tol per calcolare i valori migliori di
% theta per approssimare il dataset con una retta, con
% intercetta pari a theta(1) e coeff. ang. pari a theta(2).

function [theta] = train_lin_reg(x, y, tol)
	theta = [1. 0.1875]'; % valori di guess dei parametri da ricercare

	m = length(x); % numero di examples

	% learning factor (lo scelgo, provo)
	% ---------------
	% - se l'algoritmo mi porta a J crescenti, devo abbassare alpha;
	% - se l'algoritmo converge molto lentamente, devo alzare alpha.
	% TODO implementa un controllo su J che riduca alpha, se
	%	   J cresce troppo!
	alpha = 0.5;

	n_iter = 0;
	max_iter = 1000;
	gradJ_ = [2.*tol 2.*tol]';

	while ( ~ min_reached(gradJ_, tol) )

		if (n_iter > max_iter)
			error("max iter reached - no convergence!");
		end

		h = prediction(theta, x); % vettore (m x 1)
		diff_hy = h - y; % vettore (m x 1)

		% calcolo il gradiente di J nel punto (theta0,theta1)
		gradJ_ = gradJ(diff_hy, x);

		% calcolo del costo J, per questa scelta di theta0 e theta1
		J = costJ(diff_hy);

		% Aggiorno il valore di theta
		theta = update_theta(theta, alpha, gradJ_);

		n_iter = n_iter + 1;
	end % fine del while loop
end
