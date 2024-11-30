isSonOf(a,b).
isSonOf(b,c).
isSonOf(c,d).
isSonOf(d,e).
isSonOf(e,f).

ancestor(X,Z):-
    isSonOf(X,Z).
%Base case
%To stop execution

ancestor(X,Z) :-
    isSonOf(X,Y),
    ancestor(Y,Z).
%Recursion state



%WARNING
%DONT PUT IT LIKE THAT!!! IN ANY CASE
%ancestor(X,Z) :-
%   ancestor(Y,Z),
%   isSonOf(X,Y).


