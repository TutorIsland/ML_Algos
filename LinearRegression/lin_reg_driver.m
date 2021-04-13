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

close all;
clear all;

% mi aspetto che l'algoritmo trovi:
% theta0 = 0; theta1 = 1
%
x = [1. 2.]'; % 2 examples
y = [1. 2.]';

tol = 1e-4;
theta = train_lin_reg(x, y, tol)
