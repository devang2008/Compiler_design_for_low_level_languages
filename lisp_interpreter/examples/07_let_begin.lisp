; ============================================
; 07 - Let Bindings and Begin Blocks
; ============================================

; Let creates local scope
(let ((x 5) (y 3))
  (+ x y))

; Nested let
(let ((a 10))
  (let ((b 20))
    (* a b)))

; Begin evaluates multiple expressions, returns last
(begin
  (define counter 0)
  (define counter (+ counter 1))
  (define counter (+ counter 1))
  (define counter (+ counter 1))
  counter)

; Let with function call
(let ((radius 7)
      (pi 3.14))
  (* pi (* radius radius)))
