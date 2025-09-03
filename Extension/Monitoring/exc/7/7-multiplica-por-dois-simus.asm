ORG 0
INICIO:
    ; Este exemplo demonstra: LDA (modo direto), SHL, STA
    LDA NUMERO      ; Carrega o número (modo direto)
    SHL             ; Desloca 1 bit à esquerda (multiplica por 2)
    STA RESULTADO   ; Armazena resultado (modo direto)
    HLT             ; Para o programa

; Dados
NUMERO:    25       ; Número a ser multiplicado por 2
RESULTADO: DS 1     ; Resultado (deveria ser 50)

END 0
