ORG 0
INICIO:
    LDA #123                    ; carga imediata do valor a ser copiado
    STA VALOR_ORIGEM            ; armazena em VALOR_ORIGEM
    LDA #VALOR_ORIGEM           ; carrega o endereço de VALOR_ORIGEM
    STA PONTEIRO                ; armazena o ponteiro
    LDA @PONTEIRO               ; carrega indiretamente o valor apontado
    STA DESTINO                 ; armazena no destino (modo direto)
    HLT                         ; Para o programa

; Dados
VALOR_ORIGEM: DS 1
PONTEIRO:     DS 1
DESTINO:      DS 1          ; Destino da cópia

END 0
                             