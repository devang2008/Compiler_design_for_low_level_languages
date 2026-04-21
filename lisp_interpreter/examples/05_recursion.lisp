
; 05 - Recursion (Factorial and Fibonacci)


; Factorial
(define (factorial n)
  (if (= n 0)
    1
    (* n (factorial (- n 1)))))

(display "factorial(5) = ")
(print (factorial 5))

(display "factorial(10) = ")
(print (factorial 10))

; Fibonacci
(define (fibonacci n)
  (if (< n 2)
    n
    (+ (fibonacci (- n 1))
       (fibonacci (- n 2)))))

(display "fibonacci(10) = ")
(print (fibonacci 10))

(display "fibonacci(15) = ")
(print (fibonacci 15))

; Sum of list using recursion
(define (sum-list lst)
  (if (null? lst)
    0
    (+ (car lst) (sum-list (cdr lst)))))

(display "sum(1..5) = ")
(print (sum-list (list 1 2 3 4 5)))
