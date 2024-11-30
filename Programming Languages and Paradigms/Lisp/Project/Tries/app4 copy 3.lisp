;main
(defun main ()

   (let (  (margemA '(repolho lobo pescador cabra))
           (margemB '())
        )

        (print (exec margemA margemB (list (list margemA margemB))))

    )

)

;recursão
(defun exec (margemA margemB caminho)

    ;Chegou-se ao resultado?
    (if (and (member 'pescador margemB) (member 'lobo margemB) (member 'cabra margemB) (member 'repolho margemB) )
        
        ;Sim
        (print 'done)
       
        ;Não
        (progn

        (print margemA)
        (print margemB)
        (FRESH-LINE)
            
            ;Para cada elemento, execute
            (dolist (item (append margemA margemB))
                    
                ;Tente enviar um elemento de A para B
                ;Tranferencia é possivel?
                (if (permitido_transfer margemA margemB item caminho)
                            
                    ;Sim
                    (let*
                        (
                            (novamargemA (remove 'pescador margemA))
                            (novamargemA (remove item novamargemA))
                            (novamargemB (remove-duplicates (cons 'pescador (cons item margemB))))
                            (novocaminho (push (list novamargemA novamargemB) caminho)) 
                        )

                        ;Tranfira, adicione ao caminho e chame exec
                        (exec novamargemA novamargemB novocaminho)

                    )

                )

                ;Tente enviar um elemento de B para A
                ;Tranferencia é possivel?
                (if (permitido_transfer margemB margemA item caminho)
                            
                    ;Sim
                    (let*
                        (
                            (novamargemB (remove 'pescador margemB))
                            (novamargemB (remove item novamargemB))
                            (novamargemA (remove-duplicates (cons 'pescador (cons item margemA))))
                            (novocaminho (push (list novamargemA novamargemB) caminho)) 
                        )

                        ;Tranfira, adicione ao caminho e chame exec
                        (exec novamargemA novamargemB novocaminho)

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
    (let*
        (
            (novamargemA (remove 'pescador margemA))
            (novamargemA (remove item margemA))
            (novamargemB (remove-duplicates (cons 'pescador (cons item margemB))))
        )

        ;O item e o pescador existe na margem?
        (if (and (not (null item)) (member item margemA) (member 'pescador margemA))                            
            (progn 

                ;Posso transferir esse item para a outra margem sem combinações ilegais?    
                (if (permitido novamargemA )                            
                    (progn 
                                
                        ;mA e mB novos estão no caminho?
                        (if (member (list novamargemA novamargemB) caminho :test #'equal)
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

)

;Chamamos o programa
(main)