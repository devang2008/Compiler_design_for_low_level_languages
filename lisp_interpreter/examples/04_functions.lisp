
; 04 - Functions (define and lambda)


; Named function shorthand
(define (square x) (* x x))
(square 5)
(square 12)

; Lambda (anonymous function)
(define double (lambda (x) (* x 2)))
(double 7)

; Function that returns a function (closure!)
(define (make-adder n)
  (lambda (x) (+ n x)))

(define add10 (make-adder 10))
(add10 5)
(add10 100)

; Higher-order: map
(map square (list 1 2 3 4 5))
