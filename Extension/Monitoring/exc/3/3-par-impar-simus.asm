ORG 0
INICIO:
    LDA #42         ; carga imediata do número a ser testado
    STA NUMERO      ; armazena o número em NUMERO
    LDA NUMERO      ; carrega o número (modo direto)
    AND #1          ; Testa o bit menos significativo (modo imediato)
    JZ PAR          ; Se zero, é par (desvio condicional)

    ; É ímpar
    LDA #0          ; Carrega 0 para ímpar (modo imediato)
    STA RESULTADO   ; Armazena resultado (modo direto)
    JMP FIM         ; Pula para o fim (desvio incondicional)

PAR:
    LDA #1          ; Carrega 1 para par (modo imediato)
    STA RESULTADO   ; Armazena resultado (modo direto)

FIM:
    HLT             ; Para o programa

; Dados
NUMERO:    DS 1
RESULTADO: DS 1     ; 1=par, 0=ímpar

END 0              