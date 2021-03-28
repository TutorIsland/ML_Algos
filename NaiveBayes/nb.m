% implementazione dell'algoritmo di Naive Bayes - classificatore

% Per informazioni sul codice, sull'algoritmo di Naive Bayes,
% per supporto nei tuoi studi ingegneristici,
% contattami:
%				contact.tutorisland@gmail.com
%
% Con oltre 10 anni di esperienza nel supporto a studenti di varie facolta'
% ingegneristiche/scientifiche, posso guidarti nei tuoi studi, sia per
% la tecnica, il metodo di studio, che per capire bene gli argomenti.
%
% Offro consulenze per la tua carriera accademica e consulenze tecniche
% ingegneristiche in ambito di Modellazione Numerica.
% Contattami:
%				contact.tutorisland@gmail.com

% Questo codice e' stato sviluppato durante una diretta sul canale
% YouTube "Tutor Island":
% https://www.youtube.com/channel/UCKkzN06obaHk8mt3iBTp6qw?sub_confirmation=1
%
% Link al video con la spiegazione del codice:
% 

% in colonna ciascuna persona
%
%   1    /   0
% felice / triste
% ha finito di studiare / non ha finito di studiare
% sole / piogga
A = [1 0 0 1 1;
     1 1 0 1 0;
     0 0 1 1 0];
y = [1 0 0 1 1]; % la persona esce di casa (1) o no (0) ?


% nuovo caso di cui voglio sapere la probabilita' che la persona che si trova
% in queste condizioni uscira' di casa
x = [1;
     1;
     1];

N = columns(A);
M = rows(A);


%% CASO DI POSITIVO

c = 1; % la persona decide di uscire
N_c = sum( y == c );
p_c = N_c/N;

casi_c = ( y == c );
produttoria = 1;
A_sub = A(:,casi_c);
for k = 1 : M
	N_c_xk = sum( A_sub(k,:) == x(k) );
	N_xk = sum( A(k,:) == x(k) );
	p_c_xk = N_c_xk/N_xk;
	produttoria = produttoria * p_c_xk;
end
p_pos_prop = produttoria/(p_c^(M-1)); %%%%


%% CASO DI NEGATIVO

c = 0; % la persona decide di NON uscire
N_c = sum( y == c );
p_c = N_c/N;

casi_c = ( y == c );
produttoria = 1;
A_sub = A(:,casi_c);
for k = 1 : M
	N_c_xk = sum( A_sub(k,:) == x(k) );
	N_xk = sum( A(k,:) == x(k) );
	p_c_xk = N_c_xk/N_xk;
	produttoria = produttoria * p_c_xk;
end
p_neg_prop = produttoria/(p_c^(M-1));

tot = p_pos_prop + p_neg_prop;

p_pos = p_pos_prop / tot
p_neg = 1 - p_pos

percentuale_pos = p_pos*100;

threshold = 0.5;
printf("  ###  TUTOR ISLAND  ###\n");
printf("  L'algoritmo dice che probabilmente la persona uscira' di casa: ");
if (p_pos > threshold)
	printf("si'!\n");
else
	printf("no!\n");
end
printf(" %s%% %f\n", " La probabilita' di uscire di casa calcolata e': ", percentuale_pos);

printf("  ### per informazioni e supporto nei tuoi studi: contact.tutorisland@gmail.com\n");
