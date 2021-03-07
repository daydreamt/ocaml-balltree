TODO: For d > 1, depending on the input shape [-1;1] vs [1;-1], Tensor.dist returns a different result, so we have to choose which of the 2 formats we expect the input in, and assert everywhere accordingly.
TODO: let bad_points = Tensor.of_float2  [| [| 0.; 0. |]; [| 0.; 0. |] |];; is broken
    because of line 72
