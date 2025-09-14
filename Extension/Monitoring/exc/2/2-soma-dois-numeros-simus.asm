ORG 0
INICIO:
    LDA #15         ; carga imediata do primeiro número
    STA NUM1        ; armazena NUM1
    LDA #27         ; carga imediata do segundo número
    STA NUM2        ; armazena NUM2
    LDA NUM1        ; Carrega primeiro número (modo direto)
    ADD NUM2        ; Soma com segundo número (modo direto)
    STA RESULTADO   ; Armazena resultado (modo direto)
    HLT             ; Para o programa

; Dados
NUM1:    DS 1       ; Primeiro número (inicializado no inicio)
NUM2:    DS 1       ; Segundo número (inicializado no inicio)
RESULTADO: DS 1     ; Resultado da soma

END 0
