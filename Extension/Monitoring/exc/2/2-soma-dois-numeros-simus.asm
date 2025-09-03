ORG 0
INICIO:
    ; Este exemplo demonstra: LDA (modo direto), ADD (modo direto), STA (modo direto)
    LDA NUM1        ; Carrega primeiro número (modo direto)
    ADD NUM2        ; Soma com segundo número (modo direto)
    STA RESULTADO   ; Armazena resultado (modo direto)
    HLT             ; Para o programa

; Dados
NUM1:      15       ; Primeiro número
NUM2:      27       ; Segundo número  
RESULTADO: DS 1     ; Resultado da soma

END 0
