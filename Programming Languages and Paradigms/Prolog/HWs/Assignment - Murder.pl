%Fatos iniciais
morto(jean, terca).

suspeito(lucas).
suspeito(paulo).
suspeito(alan).
suspeito(bernardo).
suspeito(luis).


%Fatos da investigação
alibi(lucas, bernardo, terca).
alibi(paulo, bernardo, terca).
alibi(luis, lucas, terca).      % alibi(Pessoa, QuemFalou, ParaQuando)
alibi(alan, lucas, quinta).

naoConfiavel(alan).

querVinganca(paulo, jean).      % querVingança(QuemQuer, ContraQuem)
querVinganca(lucas, jean).

beneficiario(bernardo, jean).   % beneficiario(Quemé, deQuem)
beneficiario(jean, luis).

deveDinheiro(luis, jean).       % deveDinheiro(QuemDeve, ParaQuem).
deveDinheiro(lucas, jean).

viuCometerCrime(jean, alan).    % viuCometerCrime(QuemViu, QuemFez) 

possuiArma(lucas).
possuiArma(luis).
possuiArma(alan).


%Regras
confiavel(Pessoa):-
    \+(naoConfiavel(Pessoa)).

alibiValido(Pessoa, Dia):-
    alibi(Pessoa, QuemFalou, Dia),
    confiavel(QuemFalou).

possuiMotivo(Pessoa, Morto):-
    possuiInteresse(Pessoa, Morto);
    querVinganca(Pessoa, Morto).

possuiInteresse(Pessoa, Morto):-
    beneficiario(Morto, Pessoa);
    deveDinheiro(Pessoa, Morto);
    viuCometerCrime(Morto, Pessoa).

assassino(Assassino, Morto, Dia) :-
    possuiArma(Assassino),
    possuiMotivo(Assassino, Morto),
    \+(alibiValido(Assassino, Dia)).

resolver(Assassino):-
    assassino(Assassino, jean, terca).

