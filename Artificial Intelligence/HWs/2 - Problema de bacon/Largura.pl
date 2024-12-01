set_prolog_stack(global, limit(100000000000)).

%Importa a base de dados
:- consult('atores_grau2').

%Algumas funções necessárias (pertencer/concatenar/...)
pertence(E, [E|_]).
pertence(E, [_|T]) :-
    pertence(E,T).

s([X, _, _], [Y, Z, A]) :-
    ator(X, Z, A, _),
    ator(Y, Z, A, _),
    Y \= X.

final([X, _, _], NoFinal) :-
    X == NoFinal.

concatenar([],L,L).
concatenar([X|L1],L2,[X|L3]) :-
    concatenar(L1,L2,L3).

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

%Implementa a busca em largura
resolva(No, NoFinal, Solucao) :-
    breadthFirst([[No]], Solucao, NoFinal).

breadthFirst([[No|Caminho]|_],[No|Caminho], NoFinal) :-
    final(No, NoFinal).

breadthFirst([Caminho|Caminhos], Solucao, NoFinal) :-
    estender(Caminho, NovosCaminhos),
    concatenar(Caminhos, NovosCaminhos, Caminhos1),
    write(Caminho), nl,
    breadthFirst(Caminhos1, Solucao, NoFinal).

estender([No|Caminho], NovosCaminhos) :-
    findall([NovoNo, No|Caminho], (s(No,NovoNo), \+ pertence(NovoNo,[No|Caminho])), NovosCaminhos).

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


