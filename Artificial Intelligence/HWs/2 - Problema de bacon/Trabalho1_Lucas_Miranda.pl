%Trabalho 1
%Aluno: Lucas Miranda Mendonça Rezende
%Nusp:  12542838

%Importa a base de dados.
:- consult('atores_grauN').

%Algumas funcoes auxiliares (pertence/último elemento/número de elementos).
pertence(E, [E|_]).
pertence(E, [_|T]) :-
    pertence(E,T).

ultimo(Elemento, [Elemento]).
ultimo(Elemento, [_|Cauda]) :-
    ultimo(Elemento,Cauda).

count([], -1).
count([_|Tail], N) :-
    count(Tail, N1),
    N is N1 + 1.

%Define relação de nós sucessores e finais.
s([X, _, _], [Y, Z, A]) :-
    ator(X, Z, A, _),
    ator(Y, Z, A, _),
    Y \= X.

final([X, _, _], NoFinal) :-
    X == NoFinal.

%Implementa a busca em profundidade iterativa.
%Este método foi escolhido pois garante que o menor caminho será encontrado, além necessitar de menos memória considerando que o espaço de busca é muito grande.
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

%Implementa a função "Conectar".
conectar(No, NoFinal) :-
    resolva([No, _, _], NoFinal, Solucao),
    count(Solucao, N),
    nl, write("Sim. "), write(NoFinal), write(" esta separado de "), write(No), write(" por "), write(N), write(" graus."), nl,
    mostrar_lista_start(Solucao), nl.

%Implementa a função que mostra o resultado da busca conforme especificado pelo trabalho.
mostrar_lista_start([[Y, Z, A]|Resto]):-
    mostrar_lista(Resto, Y, Z, A).

mostrar_lista([], _, _, _).
mostrar_lista([[Y, Z, A]|Resto], YY, ZZ, AA) :-
    write(YY), write(" esteve em "), write(ZZ), write(" com "), write(Y), write(" em "), write(AA), write("."), nl,
    mostrar_lista(Resto, Y, Z, A).