% A partire dal dataset in input (feature x, outcome noti y),
% usa una tolleranza tol per calcolare i valori migliori di
% theta per approssimare il dataset con una retta, con
% intercetta pari a theta(1) e coeff. ang. pari a theta(2).
% Implementa l'algoritmo "Linear Regression", usando
% il "Gradient Descent" ed una funzione di costo J convessa
% quadratica.
%
% Restituisce:
% - theta: i valori theta0 e theta1 calcolati nella fase di training
% - n_iter: numero di iterazioni necessarie con l'ultimo seed testato

function [theta, n_iter] = train_lin_reg(x, y, tol)
	m = length(x); % numero di examples nel mio dataset

	% nota sul learning factor
	% ------------------------
	% - se l'algoritmo mi porta a J crescenti, devo ridurre alpha;
	% - se l'algoritmo converge molto lentamente, potrei aumentare alpha.

	[alpha theta J_prev gradJ_ max_iter alpha_min alpha_reducing_factor] = init_learning_params(tol);

	n_iter = 0; % considering possible seed resets

	while ( ~ min_reached(gradJ_, tol) )

		if (n_iter > max_iter)
			error("max iter reached - no convergence!");
		end

		h = predict(theta, x); % vettore (m x 1)
		diff_hy = h - y; % vettore (m x 1)

		% calcolo il gradiente di J nel punto (theta0,theta1)
		gradJ_ = gradJ(diff_hy, x);

		% Aggiorno il valore di theta
		theta = update_theta(theta, alpha, gradJ_);

		% calcolo del costo J, per questa scelta di theta0 e theta1
		J = costJ(diff_hy);

		% controllo sul costo dovuto a questi valori di theta0 e theta1
		%
		% se il costo e' aumentato rispetto alla precedente iterazione:
		% - se alpha e' diventata piccolissima:
		% 	- risetta alpha al suo valore iniziale
		%   - procedi con un seed diverso
		%	- risetta J_prev al massimo valore possibile
		% - altrimenti
		%   - riduci alpha
		% 	- aggiorna J_prev al J appena calcolato
		[alpha theta J_prev] = check_gradient_descent(J, J_prev, alpha, alpha_min, alpha_reducing_factor, theta);

		n_iter++;
	end % fine del while loop
end
