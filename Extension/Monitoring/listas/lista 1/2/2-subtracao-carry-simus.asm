ORG 0
INICIO:
    LDA #255        ; Carrega 255 (modo imediato - 8 bits)
    ADD #1          ; Soma 1 para gerar carry (overflow)
    
    LDA #100        ; carga imediata do primeiro número
    STA NUM1
    LDA #20         ; carga imediata do segundo número
    STA NUM2
    LDA NUM1        ; Carrega primeiro número (modo direto)
    SBC NUM2        ; Subtrai NUM2 com carry (modo direto)
    STA RESULTADO   ; Armazena resultado (modo direto)
    HLT             ; Para o programa

; Dados
NUM1:      DS 1     ; Primeiro número
NUM2:      DS 1     ; Número a ser subtraído
RESULTADO: DS 1     ; Resultado da subtração com carry

END 0
