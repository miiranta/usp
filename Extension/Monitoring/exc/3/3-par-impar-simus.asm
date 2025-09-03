ORG 0
INICIO:
    ; Este exemplo demonstra: LDA (modo direto), AND (modo imediato), JZ, JMP
    LDA NUMERO      ; Carrega o número (modo direto)
    AND #1          ; Testa o bit menos significativo (modo imediato 8 bits)
    JZ PAR          ; Se zero, é par (desvio condicional)
    
    ; É ímpar
    LDA #0          ; Carrega 0 para ímpar (modo imediato 8 bits)
    STA RESULTADO   ; Armazena resultado (modo direto)
    JMP FIM         ; Pula para o fim (desvio incondicional)
    
PAR:
    LDA #1          ; Carrega 1 para par (modo imediato 8 bits)
    STA RESULTADO   ; Armazena resultado (modo direto)
    
FIM:
    HLT             ; Para o programa

; Dados
NUMERO:    42       ; Número a ser testado
RESULTADO: DS 1     ; 1=par, 0=ímpar

END 0
