like(i, candy).
like(i, swimming).
like(you, candy).
like(i, you).

likeBoth(X, Y, Z) :-
    like(X,Y),
    like(X,Z).
%AND operator (,).

likeOneAtleast(X, Y, Z) :-
    like(X,Y);
    like(X,Z).
%OR operator (;).

likeOneAtleast(X, Y, _) :-
    like(X,Y).
likeOneAtleast(X, _, Z) :-
    like(X,Z).
%Same as OR operator (;).
%Analysis is procedural from top to bottom, things on the top have priority!

likeBothNoRepeat(X, Y, Z) :-
    like(X,Y),
    like(X,Z),
    Y \= X.
%DIFFERENCE operator (\=).

likeBothRepeat(X, Y, Z) :-
    like(X,Y),
    like(X,Z),
    Y == X.
%EQUALITY operator (==).

isSumRightEqual :-
    3 = 2 + 1.
%UNIFICATION operator (=).
%Views as 3 = +(2,1) which is false

isSumRightIs :-
    3 is 2 + 1.
%FORCE ARITHMETIC operation (is).
%Forces the math on the right, so 3 = 3 is true

isAlwaysFalse(X, Y) :-
   \+(Y = X).
%NOT operator (\+()).
%Y = X is always true, so not true is false
