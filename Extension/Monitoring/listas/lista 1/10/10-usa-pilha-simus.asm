ORG 0
INICIO:
    LDA #45         ; carga imediata do primeiro valor
    STA VALOR_A
    LDA VALOR_A     ; Carrega VALOR_A (modo direto)
    PUSH            ; Empilha VALOR_A

    LDA #35         ; carga imediata do segundo valor
    STA VALOR_B
    LDA VALOR_B     ; Carrega VALOR_B (modo direto)
    STA TEMP        ; Salva VALOR_B temporariamente
    
    POP             ; Desempilha VALOR_A para o acumulador
    ADD TEMP        ; Soma com VALOR_B
    STA RESULTADO   ; Armazena resultado (modo direto)
    HLT             ; Para o programa

; Dados
VALOR_A:   DS 1     ; Primeiro valor
VALOR_B:   DS 1     ; Segundo valor
TEMP:      DS 1     ; Variável temporária
RESULTADO: DS 1     ; Resultado da soma

END 0
