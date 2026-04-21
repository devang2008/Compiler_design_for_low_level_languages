
; 03 - Conditionals and Boolean Logic


(if (> 10 5) "ten is bigger" "five is bigger")

(define age 18)
(if (>= age 18) "adult" "minor")

; Boolean operators
(and #t #t)
(and #t #f)
(or #f #t)
(not #f)

; Comparisons
(= 5 5)
(< 3 7)
(> 10 2)

; Cond (multi-branch)
(define score 85)
(cond
  ((>= score 90) "A")
  ((>= score 80) "B")
  ((>= score 70) "C")
  (else "F"))
