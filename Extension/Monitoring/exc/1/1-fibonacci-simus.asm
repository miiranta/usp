ORG 0
INICIO:
    LDA #0       ; FIB1 = 0
    STA FIB1
    LDA #1       ; FIB2 = 1
    STA FIB2

LOOP:
    ; Calcula próximo Fibonacci
    LDA FIB1
    ADD FIB2
    STA FIBNEXT

    ; Atualiza FIB1 e FIB2
    LDA FIB2
    STA FIB1
    LDA FIBNEXT
    STA FIB2

    JMP LOOP     ; repete indefinidamente

; memória
FIB1:    DS 1
FIB2:    DS 1
FIBNEXT: DS 1

END 0

