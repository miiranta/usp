ORG 0
INICIO:
    ; Este exemplo demonstra: LDA (modo imediato), SUB (modo imediato), JN, ADD
    LDA #1          ; Inicializa contador com 1 (modo imediato 8 bits)
    STA CONTADOR    ; Armazena contador (modo direto)

LOOP:
    LDA CONTADOR    ; Carrega contador atual (modo direto)
    SUB #5          ; Subtrai 5 (modo imediato 8 bits)
    JN CONTINUA     ; Se negativo, continua (contador < 5)
    JZ CONTINUA     ; Se zero, continua (contador = 5)
    JMP FIM         ; Se positivo, termina (contador > 5)

CONTINUA:
    LDA CONTADOR    ; Carrega contador atual (modo direto)
    ADD #1          ; Incrementa contador (modo imediato 8 bits)
    STA CONTADOR    ; Armazena novo valor (modo direto)
    JMP LOOP        ; Volta para o loop (desvio incondicional)

FIM:
    HLT             ; Para o programa

; Dados
CONTADOR: DS 1      ; Contador atual

END 0
