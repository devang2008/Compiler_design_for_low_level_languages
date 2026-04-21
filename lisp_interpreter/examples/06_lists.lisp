; 06 - List Operations

(define mylist (list 1 2 3 4 5))

; car = first element
(display "car: ")
(print (car mylist))

; cdr = rest of list
(display "cdr: ")
(print (cdr mylist))

; cons = prepend
(display "cons 0: ")
(print (cons 0 mylist))

; length
(display "length: ")
(print (length mylist))

; null check
(display "null? empty: ")
(print (null? (list)))

(display "null? non-empty: ")
(print (null? mylist))

; map and filter
(display "doubled: ")
(print (map (lambda (x) (* x 2)) mylist))

(display "evens: ")
(print (filter (lambda (x) (= (modulo x 2) 0)) mylist))
