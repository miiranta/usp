;Main
(defun main ()

    (let (  (margemA '(repolho cabra lobo pescador))
            (margemB '())
         )

        (exec margemA margemB margemA margemB)

    )
)

;Definimos uma função para chamar a recursão
(defun exec (margemA margemB listA listB)

    ;Chegou-se ao resultado?
    (if (and (member 'pescador margemB) (member 'lobo margemB) (member 'cabra margemB) (member 'repolho margemB) )
        (progn

            ;Print
            (print margemA)
            (print margemB)
            (FRESH-LINE)
        )
    )


        (dolist (item listA)
        
            ;Transferir A->B
            (if (member 'pescador margemA)
                (progn
                    (setf margemA (remove 'pescador margemA)) 
                    (setf margemA (remove item margemA)) 
                    (setf margemB (remove-duplicates (cons 'pescador (cons item margemB))) ) 

                    ;Print
                    (print margemA)
                    (print margemB)
                    (FRESH-LINE)

                    ;Tranferencia foi permitida?
                    (if (and (permitido margemA) (permitido margemB) ) 
                            
                        ;Sim
                        (progn
                            (print 'a)
                            (exec margemA margemB (remove item listA) listB)
                        )
                    
                        ;Não
                        (progn
                            (print 'b)
                            (exec margemA margemB (list item) (list item))
                        )
                        

                    )  
                )
            )
        
            
        
        )

        (dolist (item listB)
        
            ;Transferir B->A
            (if (member 'pescador margemB)
                (progn
                    (setf margemB (remove 'pescador margemB)) 
                    (setf margemB (remove item margemB)) 
                    (setf margemA (remove-duplicates (cons 'pescador (cons item margemA))) ) 

                    ;Print
                    (print margemA)
                    (print margemB)
                    (FRESH-LINE)

                    ;Tranferencia foi permitida?
                    (if (and (permitido margemA) (permitido margemB)) 
                            
                        ;Sim
                        (progn
                            (print 'a)
                            (exec margemA margemB margemA (remove item margemB))
                        )

                        ;Não
                        (progn
                            (print 'd)
                            (exec margemA margemB (list item) (list item))
                        )
                    
                        

                    )  


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

;Inicia o programa
(main)
