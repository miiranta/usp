ORG 0
INICIO:
    ; Este exemplo demonstra: LDA, ADD (para gerar carry), SBC
    LDA #255        ; Carrega 255 (modo imediato 8 bits)
    ADD #1          ; Soma 1 para gerar carry (overflow)
    
    LDA NUM1        ; Carrega primeiro número (modo direto)
    SBC NUM2        ; Subtrai NUM2 com carry (modo direto)
    STA RESULTADO   ; Armazena resultado (modo direto)
    HLT             ; Para o programa

; Dados
NUM1:      100      ; Primeiro número
NUM2:      30       ; Número a ser subtraído
RESULTADO: DS 1     ; Resultado da subtração com carry

END 0
