ORG 0
INICIO:
    ; Este exemplo demonstra: LDA, PUSH, POP, ADD
    LDA VALOR_A     ; Carrega VALOR_A (modo direto)
    PUSH            ; Empilha VALOR_A
    
    LDA VALOR_B     ; Carrega VALOR_B (modo direto)
    STA TEMP        ; Salva VALOR_B temporariamente
    
    POP             ; Desempilha VALOR_A para o acumulador
    ADD TEMP        ; Soma com VALOR_B
    STA RESULTADO   ; Armazena resultado (modo direto)
    HLT             ; Para o programa

; Dados
VALOR_A:   45       ; Primeiro valor
VALOR_B:   35       ; Segundo valor
TEMP:      DS 1     ; Variável temporária
RESULTADO: DS 1     ; Resultado da soma

END 0
