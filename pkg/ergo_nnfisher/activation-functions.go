package ergo_nnfisher

import "math"

// ActivationSigmoid является реализацией сигмоиды,
// используемой для активации.
func ActivationSigmoid(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-1.0*v))
}

func SygmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}


// SigmoidBack является реализацией производной
// сигмоиды для обратного распространения.
func SigmoidBack(x float64) float64 {
	return x * (1.0 - x)
}
