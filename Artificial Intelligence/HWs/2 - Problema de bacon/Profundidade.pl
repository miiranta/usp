%Importa a base de dados
:- consult('atores_grauN').

%Algumas funções auxiliares
pertence(E, [E|_]).
pertence(E, [_|T]) :-
    pertence(E,T).

ultimo(Elemento, [Elemento]).
ultimo(Elemento, [_|Cauda]) :-
    ultimo(Elemento,Cauda).

deleta_ultimo(X, Y) :-
    reverse(X, [_|X1]), 
    reverse(X1, Y).

count([], -1).
count([_|Tail], N) :-
    count(Tail, N1),
    N is N1 + 1.

s([X, _, _], [Y, Z, A]) :-
    ator(X, Z, A, _),
    ator(Y, Z, A, _),
    Y \= X.

final([X, _, _], NoFinal) :-
    X == NoFinal.

%Implementa a busca em altura
resolva(No, NoFinal, Solucao) :-
    depthFirstIterativeDeepening(No, Solucao, NoFinal).

path(No, No, [No]). 
path(Primeiro, Ultimo, [Ultimo|Caminho]) :-
    path(Primeiro, Penultimo, Caminho),
    s(Penultimo, Ultimo),
    \+ pertence(Ultimo, Caminho).
     
depthFirstIterativeDeepening(No, Solucao, NoFinal) :-
    path(No, Final, Solucao),
    final(Final, NoFinal).

%Conectar
conectar(No, NoFinal) :-
    resolva([No, _, _], NoFinal, Solucao),
    count(Solucao, N),
    nl, write("Sim. "), write(NoFinal), write(" esta separado de "), write(No), write(" por "), write(N), write(" graus."), nl,
    deleta_ultimo(Solucao, NovaSolucao),
    mostrar_lista(NovaSolucao, No), nl.

mostrar_lista([], _).
mostrar_lista(Lista, X) :-
    ultimo([Y, Z, A], Lista),
    write(X), write(" esteve em "), write(Z), write(" com "), write(Y), write(" em "), write(A), write("."), nl,
    deleta_ultimo(Lista, NovaLista),
    mostrar_lista(NovaLista, Y).