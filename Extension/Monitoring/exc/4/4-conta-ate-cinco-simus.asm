ORG 0
INICIO:
    LDA #1          ; Inicializa contador com 1 (modo imediato)
    STA CONTADOR    ; Armazena contador (modo direto)

LOOP:
    LDA CONTADOR    ; Carrega contador atual (modo direto)
    SUB #5          ; Subtrai 5 (modo imediato)
    JN CONTINUA     ; Se negativo, continua (contador < 5)
    JZ FIM          ; Se zero, termina (contador = 5)
    JMP FIM         ; Se positivo, termina (contador > 5)

CONTINUA:
    LDA CONTADOR    ; Carrega contador atual (modo direto)
    ADD #1          ; Incrementa contador (modo imediato)
    STA CONTADOR    ; Armazena novo valor (modo direto)
    JMP LOOP        ; Volta para o loop (desvio incondicional)

FIM:
    HLT             ; Para o programa

; Dados
CONTADOR: DS 1      ; Contador atual

END 0
