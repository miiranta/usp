ORG 0
INICIO:
    ; Inicializa os dois primeiros termos da sequência
    LDA #0        ; carga imediata 0 (F(0))
    STA FIB1      ; armazena em FIB1
    LDA #1        ; carga imediata 1 (F(1))
    STA FIB2      ; armazena em FIB2

LOOP:
    ; Calcula próximo termo: FIBNEXT = FIB1 + FIB2
    LDA FIB1      ; carrega F(n-1)
    ADD FIB2      ; soma com F(n)
    STA FIBNEXT   ; armazena temporariamente F(n+1)

    ; Atualiza registros em memória: F(n) passa a ser F(n-1), F(n+1) passa a ser F(n)
    LDA FIB2      ; carrega F(n)
    STA FIB1      ; armazena em FIB1 (agora é F(n-1))
    LDA FIBNEXT   ; carrega F(n+1)
    STA FIB2      ; armazena em FIB2 (agora é F(n))

    JMP LOOP      ; repete

; Dados
FIB1:    DS 1       ; termo anterior
FIB2:    DS 1       ; termo atual
FIBNEXT: DS 1       ; próximo termo (temporário)

END 0