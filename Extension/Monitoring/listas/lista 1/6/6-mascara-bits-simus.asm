ORG 0
INICIO:
    LDA #170        ; carga imediata do número original (10101010b)
    STA NUMERO
    LDA #15         ; carga imediata da máscara (00001111b)
    STA MASCARA
    LDA NUMERO      ; Carrega o número (modo direto)
    AND MASCARA     ; Aplica máscara bit a bit (modo direto)
    STA RESULTADO   ; Armazena resultado (modo direto)
    HLT             ; Para o programa

; Dados
NUMERO:    DS 1     
MASCARA:   DS 1     
RESULTADO: DS 1     ; Resultado da operação AND (10101010 AND 00001111 = 00001010 ou 10 em decimal)

END 0
