ORG 0
INICIO:
    ; Este exemplo demonstra: LDA (modo direto), AND (modo direto), STA
    LDA NUMERO      ; Carrega o número (modo direto)
    AND MASCARA     ; Aplica máscara bit a bit (modo direto)
    STA RESULTADO   ; Armazena resultado (modo direto)
    HLT             ; Para o programa

; Dados
NUMERO:    170      ; Número original (10101010 em binário)
MASCARA:   15       ; Máscara (00001111 em binário)
RESULTADO: DS 1     ; Resultado da operação AND

END 0
