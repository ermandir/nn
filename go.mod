module github.com/ermandir/nn

go 1.17

replace github.com/ermandir/nn/pkg/ergo_mnist => ./pkg/ergo_mnist

replace github.com/ermandir/nn/pkg/ergo_nnfisher => ./pkg/ergo_nnfisher

require gonum.org/v1/gonum v0.9.3

require golang.org/x/exp v0.0.0-20200224162631-6cc2880d07d6 // indirect
