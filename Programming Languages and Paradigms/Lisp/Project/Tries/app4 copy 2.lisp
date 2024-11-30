;main
(defun main ()

   (let (  (margemA '(repolho lobo pescador cabra))
           (margemB '())
        )

        (exec margemA margemB () margemA)

    )

)

;recursão
(defun exec (margemA margemB caminho qlist)

    ;Chegou-se ao resultado?
    (if (and (member 'pescador margemB) (member 'lobo margemB) (member 'cabra margemB) (member 'repolho margemB) )
        
        ;Sim
        (progn
            ;Print
            (print margemA)
            (print margemB)
            (FRESH-LINE)
        )

        (progn

            (print caminho)
            (FRESH-LINE)
        
            ;A lista de testes está vazia?
            (if (null qlist)

                ;Sim
                (exec margemA margemB caminho (append margemA margemB))

                ;Não
                (progn
                
                    ;Para cada elemento, execute
                    (dolist (item qlist)
                    
                        ;Tente enviar um elemento de A para B
                        ;Tranferencia é possivel?
                        (if (permitido_transfer margemA margemB item caminho)
                            
                            ;Sim
                            (let*
                                (
                                    (novamargemA (remove item margemA))
                                    (novamargemB (cons item margemB))
                                    (novocaminho (append (list (list novamargemA novamargemB)) caminho)) 
                                )

                                ;Tranfira, adicione a path e chame exec com o qlist sem o item
                                (exec novamargemA novamargemB novocaminho (cdr qlist))

                            )

                        )

                        ;Tente enviar um elemento de B para A
                        ;Tranferencia é possivel?
                        (if (permitido_transfer margemB margemA item caminho)
                            
                            ;Sim
                            (let*
                                (
                                    (novamargemA (cons item margemA))
                                    (novamargemB (remove item margemB))
                                    (novocaminho (append (list novamargemA novamargemB) caminho)) 
                                )

                                ;Tranfira, adicione a path e chame exec com o qlist sem o item
                                (exec novamargemA novamargemB novocaminho (cdr qlist))

                            )

                        )


                    
                    )
                
                
                )
        
            )
        
        
        )
        

    )

    

    
    

        
    


)

;Definimos uma função para testar se há alguma combinação ilegal na margem dada
(defun permitido (margem)
    (progn

        ;A margem está vazia?
        (if (null margem)
            t
            (progn

                ;Há apenas 1 item?
                (if (null (cdr margem))
                    t
                    (progn

                        ;O pescador está na margem?
                        (if (member 'pescador margem)
                            t
                            (progn 

                                ;Há itens ilegais juntos?
                                (if (and (member 'repolho margem) (member 'cabra margem))                            
                                    nil
                                    (progn

                                        (if (and (member 'cabra margem) (member 'lobo margem))
                                            nil
                                            t
                                        )
                        
                                    )
                                )

                            )
                        )
                        
                    )
                )

            )
        )

    )
)

;Definimos uma função para saber se a transferência de um item é permitida
(defun permitido_transfer (margemA margemB item caminho)
    (progn

        ;O item e o pescador existe na margem?
        (if (and (not (null item)) (member item margemA) (member 'pescador margemA))                            
            (progn 

                ;Posso transferir esse item para a outra margem sem combinações ilegais?
                (if (permitido (cons item margemB))                            
                    (progn 
                        
                        (if (permitido (remove item margemA) )                            
                            (progn 
                                
                                ;mA e mB novos estão no caminho?
                                (if (member (list (remove item margemA) (cons item margemB)) caminho)
                                    nil
                                    t
 
                                )

                            )
                            nil
                        )

                    )
                    nil
                )
            
            )
            nil
        )

    )

)

;Chamamos o programa
(main)