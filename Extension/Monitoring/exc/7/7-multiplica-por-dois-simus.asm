ORG 0
INICIO:
    LDA #25         ; carga imediata do número a ser multiplicado
    STA NUMERO
    LDA NUMERO      ; Carrega o número (modo direto)
    SHL             ; Desloca 1 bit à esquerda (multiplica por 2)
    STA RESULTADO   ; Armazena resultado (modo direto)
    HLT             ; Para o programa

; Dados
NUMERO:    DS 1     ; Número a ser multiplicado por 2
RESULTADO: DS 1     ; Resultado (deveria ser 50)

END 0
