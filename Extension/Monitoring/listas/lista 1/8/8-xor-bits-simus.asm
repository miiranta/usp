ORG 0
INICIO:
    LDA #85            ; carga imediata do número original (01010101)
    STA NUMERO
    LDA #170           ; carga imediata da máscara XOR (10101010)
    STA MASCARA_XOR
    LDA NUMERO         ; Carrega o número (modo direto)
    XOR MASCARA_XOR    ; Aplica XOR com a máscara (modo direto)
    STA RESULTADO      ; Armazena resultado (modo direto)
    HLT                ; Para o programa

; Dados
NUMERO:     DS 1       ; Número original
MASCARA_XOR: DS 1      ; Máscara XOR
RESULTADO:  DS 1       ; Resultado do XOR (deve ser 11111111 ou 255 em decimal)

END 0                  