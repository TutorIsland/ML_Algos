% Driver della regressione lineare in MATLAB
%
% (Linear Regression)
%
% Scritto da:
% contact.tutorisland@gmail.com
%
% Canale YouTube: "Tutor Island"
%
% data: 13/04/2021
%
% Per conoscere le basi della programmazione in MATLAB,
% guarda la playlist "MATLAB per Ingegneri" sul canale
% "Tutor Island".
%
% Iscriviti a Tutor Island:
% https://www.youtube.com/channel/UCKkzN06obaHk8mt3iBTp6qw?sub_confirmation=1
%
% Aggiornamenti
% -------------
%
% 14/4/21, Tutor Island
% - aggiunto controllo sul costo: se aumenta, da una iterazione
%   alla successiva, alpha viene dimezzato
%   - se alpha diventa troppo piccolo:
%	  - viene risettato
%	  - si procede con una valori random per theta
% 15/4/21, Tutor Island
% - refactor del codice da procedurale, ad oggetti: classe Linear_Regressor

close all;
%clear all;

% mi aspetto che l'algoritmo trovi:
% theta0 = 0; theta1 = 1

x = [1. 2.]'; % 2 examples
y = [1. 2.]';

tol = 1e-4;
lr = Linear_Regressor(tol);
[theta, n_iter] = lr.train_lin_reg(x, y);
% TODO implement some fancy plotting function and call it here
