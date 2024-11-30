    eat(duck, bread).   
    eat(duck, seed).
    eat(duck, worm).
    eat(human, duck). 
    eatAlternative(i, apple).

    strong(human).
    strong(duck).
    strong(worm).
    weak(bread).
    weak(seed).

%Lets set a new relation using a rule

    isEatenBy(X,Y) :- eat(Y,X).
    % :- means IF (A true IF B true)
    %good practice: RELATIONS with lower case, VARIABLES with caps

    isEatenBy(X,Y) :- eatAlternative(Y, X).
    %"Overcharging" if the first one fails, it will try the second

    areWeak(X,Y) :- weak(X), weak(Y).
    %"Compose" rule, with AND operator (,).

    test(X,Y) :- areWeak(X,_), weak(Y).
    %"anonymous variable" (_) when the value is not useful.
    %Can be used to hide the value of some variable when asking a question too

