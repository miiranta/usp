ORG 0
INICIO:
    ; Este exemplo demonstra: LDA (modo indireto), STA (modo direto)
    LDA @PONTEIRO   ; Carrega valor apontado por PONTEIRO (modo indireto)
    STA DESTINO     ; Armazena no destino (modo direto)
    HLT             ; Para o programa

; Dados
VALOR_ORIGEM: 123   ; Valor a ser copiado
PONTEIRO:  VALOR_ORIGEM  ; Ponteiro para o valor origem
DESTINO:   DS 1     ; Destino da cópia

END 0
