ORG 0
INICIO:
    ; Este exemplo demonstra: LDA (modo direto), XOR (modo direto), STA
    LDA NUMERO         ; Carrega o número (modo direto)
    XOR MASCARA_XOR    ; Aplica XOR com a máscara (modo direto)
    STA RESULTADO      ; Armazena resultado (modo direto)
    HLT                ; Para o programa

; Dados
NUMERO:     85         ; Número original (01010101 em binário)
MASCARA_XOR: 170       ; Máscara XOR (10101010 em binário)
RESULTADO:  DS 1       ; Resultado do XOR (deveria ser 255)

END 0
