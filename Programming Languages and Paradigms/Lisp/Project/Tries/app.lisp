;Main
(defun main ()
    
    (let (  (margemA '(repolho cabra lobo))
            (margemB '())
         )

        (print margemA)
        (print margemB)
        (FRESH-LINE)

        (print (recursao margemA margemB margemA))
    )

)

;O fazendeiro deve passar todos os objetos da margem A para B
(defun recursao (margemA margemB item)
    (progn

        ;A transferencia é valida?
        (if (transferir margemA margemB (car item))
            
            ;Se sim, transfira
            (progn
                (setf margemB (transferir margemA margemB (car item)))
                (setf margemA (remove (car item) margemA))

                (print margemA)
                (print margemB)
                (FRESH-LINE)
            )

        )

        ;Lista de itens está vazia?
        (if (null (cdr item))
                    
            ;Sim
            (progn

                ;Existe apenas 1 item na margemA?
                (if (and (null (cdr margemA)) (car margemA))
                    
                    ;Sim
                    (let ( (margens (cons (car margemA) margemB)) )
                    
                        ;O problema foi resolvido!
                        (print '())
                        margens
                    
                    )

                    ;Não
                    (progn

                        ;Volta com o item para margem A
                        (setf item margemB)
                        (setf margemA (append item margemA) )
                        (setf margemB (remove (car item) margemB) )

                        (print margemA)
                        (print margemB)
                        (FRESH-LINE)

                        (setf item (remove (car item) margemA))
                        (recursao margemA margemB item)
                        
                    )
                
                )
    
            )

            ;Não, execute o próximo
            (progn
                (recursao margemA margemB (cdr item))
            )
            
            

        )

    )
)

;Definimos uma função para testar se há alguma combinação ilegal nas margens
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

;Definimos uma função para transferir um item de uma margem a outra, se não causar combinações ilegais
(defun transferir (margemA margemB item)
    (progn
    
        ;O item existe na margem?
        (if (and (member item margemA) (not (null item)) )                            
            (progn 

                ;Posso transferir esse item para a outra margem sem combinações ilegais?
                (if (permitido (cons item margemB))                            
                    (progn 
                        
                        (if (permitido (remove item margemA) )                            
                            (progn 
                                (append margemB (list item))
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

;Chamando a função main
(main)