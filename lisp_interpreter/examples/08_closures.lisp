
; 08 - Closures (Advanced)


; A counter using closures
(define (make-counter)
  (let ((count 0))
    (lambda ()
      (begin
        (set! count (+ count 1))
        count))))

(define c1 (make-counter))
(display "counter: ") (print (c1))
(display "counter: ") (print (c1))
(display "counter: ") (print (c1))

; Independent counter
(define c2 (make-counter))
(display "new counter: ") (print (c2))

; Closure captures environment
(define (make-multiplier factor)
  (lambda (x) (* factor x)))

(define triple (make-multiplier 3))
(define times10 (make-multiplier 10))

(display "triple 7 = ") (print (triple 7))
(display "times10 7 = ") (print (times10 7))
