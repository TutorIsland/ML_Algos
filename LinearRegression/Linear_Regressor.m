classdef Linear_Regressor < handle
    %LINEAR_REGRESSOR Performs Linear Regression over a dataset
    %   Given the dataset with examples and known outcomes,
    %   performs the training, learning the parameter theta.
    %   Can perform prediction, 
    %
    %   TODO implement plotting facilities (for Gradient Descent, Predictions,
    %   raw dataset, new points, ...)
    %
    %   TODO update the computations so that the regressors will also work
    %   on datasets with more than 1 feature - predict is ready for that
    
    % costanti, settate durante l'inizializzazione dell'oggetto dal
    % constructor
    properties (SetAccess=immutable)
        tol;
    end
    
    % this is how I declare C-like 'const's properties which are also private
    properties (SetAccess=immutable,GetAccess=private)
       max_iter = 10000;
       alpha_min = 1e-8;
       alpha_reducing_factor = 2.;
    end
    
    properties (Access=private)
        x (:,1) double {mustBeReal, mustBeFinite};
        y (:,1) double {mustBeReal, mustBeFinite};
        n_data {mustBeFinite, mustBeInteger};
    end
    
    % can only be set from within class (not even subclass)
    % can be read from anywhere
    properties (SetAccess=private,GetAccess=public)
        % the solution
        theta (:,1) double {mustBeFinite, mustBeReal};
        % number of iterations performed to reach the solution
        n_iter (1,1) {mustBeFinite, mustBeInteger};
    end
    
    methods
        % implement setters and getters to code specific behaviour
        % e.g. to implement lazy instantiation (getter), which means
        %      "do not instantiate until requested"
        %
        function theta = get.theta(obj)
%             disp('someone requested theta');
            theta = obj.theta;
        end
        function set.theta(this, t) % the instance can be called 'whatever'
%             disp('theta was set');
            this.theta = t;
        end
    end

    methods (Access=public)
        function obj = Linear_Regressor(tol)
            %LINEAR_REGRESSOR Construct an instance of this class
            %   Detailed explanation goes here
            obj.tol = tol;
        end

        function [theta, n_iter] = learn(obj, x, y)
            %LEARN Learn using Linear Regression
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
            
            % nota sul learning factor
            % ------------------------
            % - se l'algoritmo mi porta a J crescenti, devo ridurre alpha;
            % - se l'algoritmo converge molto lentamente, potrei aumentare alpha.

            obj.x = x;
            obj.y = y;
            obj.n_data = length(x);
            
            [alpha, theta, J_prev, gradJ_] = obj.init_learning_params();

            n_iter = 0; % considering possible seed resets

            while ( ~ obj.min_reached(gradJ_) )

                if (n_iter > obj.max_iter)
                    error("max iter reached - no convergence!");
                end

                h = obj.predict(theta, x); % vettore (m x 1)
                diff_hy = h - y; % vettore (m x 1)

                % calcolo il gradiente di J nel punto (theta0,theta1)
                gradJ_ = obj.gradJ(diff_hy, x);

                % Aggiorno il valore di theta
                theta = obj.update_theta(theta, alpha, gradJ_);

                % calcolo del costo J, per questa scelta di theta0 e theta1
                J = obj.costJ(diff_hy);

                [alpha, theta, J_prev, cont] = obj.check_gradient_descent(J, J_prev, alpha, theta);
                if (cont)
                    continue;
                end

                n_iter = n_iter + 1;
            end
            obj.theta = theta;
            obj.n_iter = n_iter;
        end
    end
    
    methods (Access=private) 
        function [J] = costJ(~, diff_hy)
            %COSTJ Compute cost
            m = length(diff_hy);
            J = ( sum(diff_hy) ).^2/(2.*m);
        end
        
        function [theta_] = update_theta(~, theta, alpha, gradJ_)
            %UPDATE_THETA Update value for parameters learnt
            theta_ = zeros(2, 1);
            theta_(1) = theta(1) - alpha*gradJ_(1);
            theta_(2) = theta(2) - alpha*gradJ_(2);
        end

        function [gJ] = gradJ(~, diff_hy, x)
            %GRADJ Compute gradient of convex cost function J
            gJ = zeros(2,1);
            m = length(diff_hy);
            gJ(1) = sum(diff_hy)/m;
            gJ(2) = (diff_hy'*x)/m;
        end        
        
        function [h] = predict(~, theta, x)
            %PREDICT Predict outcomes of x
            % Calcola il valore delle prediction,
            % dati i valori di theta (intercetta e coeff. ang.)
            % e degli examples x.
            % Vectorized computation. Returns an array of predctions,
            % one for each element of x. x can be a matrix, if more than
            % one feature is present in the dataset. TODO update the
            % remaining code!
            %
            % La function si aspetta che x sia un vettore colonna di m elementi
            % e theta sia un vettore colonna di 2 elementi [theta0 theta1]'.
            %
            m = length(x);
            % aggiungo una colonna di m "uni" al vettore x
            xx = [ones(m,1) x];

            h = xx*theta;
        end

        function [yn] = min_reached(obj, gradJ_)
            %MIN_REACHED Check if minimum has been reached
            % Restituisce vero solo se entrambe le componenti del gradiente
            % di J sono < tol; altrimenti restituisce falso.
            yn = (abs(gradJ_(1)) < obj.tol) && (abs(gradJ_(2)) < obj.tol);
        end

        % TODO turn these params into properties
        function [alpha, theta, J_prev, gradJ_] = init_learning_params(obj)
            %INIT_LEARNING_PARAMS Initialize parameters
            % Initializes the value for
            % alpha: learning rate used during Gradient Descent
            % theta: (2x1) array containing guess for theta0 and theta1
            % J_prev: cost of previous iteration set to max value possible
            alpha = 15.;
            theta = rand(2,1);
            % setta J_prev al valore piu' grande rappresentabile
            J_prev = realmax;
            % setta gradiente di J ad un valore suff. grande per entrare
            % nel while loop di learn
            gradJ_ = [2.*obj.tol 2.*obj.tol]';
        end

        function [alpha, theta, J_prev, cont] = check_gradient_descent(obj, J, J_prev, alpha, theta)
            %CHECK_GRADIENT_DESCENT Check that cost J is decreasing
            %   Checks whether gradient descent is actually happening:
            %   if cost J is increasing, instead of decreasing (as desired for proper gradient descent),
            %   learning rate is reduced.
            %   If learning rate is already small enough,
            %   learning parameters are initialized and code continues to next iteration
            %   of the while loop.
            cont = false;
            if (J > J_prev)
                if (alpha < obj.alpha_min)
                    [alpha, theta, J_prev] = obj.init_learning_params();
                    cont = true;
                    return;
                else
                    alpha = alpha/obj.alpha_reducing_factor;
                end
            end
            J_prev = J;
        end
    end
end