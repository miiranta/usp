(setf novamargemA '(b))
(setf novamargemB '(a))

(setf novocaminho ())
(setf novocaminho (push (list novamargemA novamargemB) novocaminho ))
(setf novocaminho (push (list novamargemA novamargemB) novocaminho ))

(defun sortList (elem)
    (sort (copy-list elem)  #'string-lessp :key #'first)
)

(print novocaminho)
(setf novocaminho (mapcar #'sortList novocaminho))
(print novocaminho)

(print 
    (if (member (list '(a) '(b)) novocaminho :test #'equal)
        (print "Pertence")
        (print "Não pertence")
    )
)



;(print novocaminho)
;(print (list novamargemA novamargemB))
