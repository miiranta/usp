;Main
(defun main ()

    (let (  (margemA '(repolho cabra lobo pescador))
            (margemB '())
         )

        (exec margemA margemB margemA ())

    )
)

;Definimos uma função para chamar a recursão
(defun exec (margemA margemB qlist pathlist)

    ;Chegou-se ao resultado?
    (if (and (member 'pescador margemB) (member 'lobo margemB) (member 'cabra margemB) (member 'repolho margemB) )
        (progn

            ;Print
            (print "Resultado")
            (print margemA)
            (print margemB)
            (FRESH-LINE)
        )
    )

    ;Temp vars
    (let (
            (temp1A margemA)
            (temp1B margemB)
            (temp2A margemA)
            (temp2B margemB)
         )


        ;Para cada elemento de qlist
        (dolist (item qlist)

            (setf temp1A margemA)
            (setf temp1B margemB)
        
            ;Transferir A->B
            (if (member 'pescador margemA)
                (progn
                    (setf temp1A (remove 'pescador temp1A)) 
                    (setf temp1A (remove item temp1A)) 

                    (setf temp1B (remove-duplicates (append (append temp1B (list item)) (list 'pescador) ) )) 

                
                    ;Tranferencia foi permitida?
                    (if (and (permitido temp1A) (permitido temp1B)) 
                            
                        ;Sim
                        (progn
                            (print 'a)

                            ;Print
                            (print temp1A)
                            (print temp1B)
                            (FRESH-LINE)

                            (setf pathlist item)

                            (exec temp1A temp1B qlist pathlist)
                        )
                    
                        ;Não
                        ()

                    )  
                )
            )
        
            
        
        )

        ;Para cada elemento de qlist
        (dolist (item qlist)

            (setf temp2A margemA)
            (setf temp2B margemB)
        
            ;Transferir B->A
            (if (member 'pescador margemB)
                (progn
                    (setf temp2B (remove 'pescador temp2B)) 
                    (setf temp2B (remove item temp2B)) 
                    (setf temp2A (remove-duplicates (append (append temp2A (list item)) (list 'pescador) ) )) 

                
                    ;Tranferencia foi permitida?
                    (if (and (permitido temp2A) (permitido temp2B)) 
                            
                        ;Sim
                        (progn
                            (print 'b)

                            ;Print
                            (print temp2A)
                            (print temp2B)
                            (FRESH-LINE)

                            (setf pathlist item)

                            (exec temp2A temp2B qlist pathlist)
                        )
                    
                        ;Não
                        ()

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
