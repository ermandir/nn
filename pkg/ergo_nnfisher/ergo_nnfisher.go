package ergo_nnfisher

import (
	"errors"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

var activationFunc = func(_, _ int, v float64) float64 { return ActivationSigmoid(v) }
var sigmoidBack = func(_, _ int, v float64) float64 { return SigmoidBack(v) }

// NeuralNetConfig определяет архитектуру
// и параметры обучения нашей сети.
type NeuralNetConfig struct {

	// Кол-во узлов во входном, скрытом и выходном слое
	InputNodes  int
	HiddenNodes int
	OutputNodes int

	// кол-во итераций обучения
	NumEpochs int

	//Коэффициент обучения
	LearningRate float64

	//Коэффициент весов
	WeightCoef float64
}

// neuralNetwork содержит всю информацию,
// которая определяет обученную сеть.
type NeuralNetwork struct {
	Config                       NeuralNetConfig
	weightsInputHidden           *mat.Dense
	weightsHiddenOutput          *mat.Dense
	ActivationFunction           func(i, j int, v float64) float64
}

// NewNetwork инициализирует новую нейронную сеть.
func NewNetwork(config NeuralNetConfig) *NeuralNetwork {
	nn := &NeuralNetwork{Config: config}
	nn.ActivationFunction = activationFunc // Default Activation function

	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)
	normalDispath := func(mean float64, stdDev float64) float64 {
		normalDistributed := randGen.NormFloat64()
		randWeight := normalDistributed * stdDev
		return randWeight
	}

	// Инициализируем смещения/веса.
	//todoer Сменить рандомизацию на нормальное распределение.
	//Для этого весовые коэффициен­ты выбираются из нормального
	// распределения с центром в нуле и со стандартным отклонением,
	// величина которого обратно пропорцио­нальна корню
	// квадратному из количества входящих связей на узел.
	// self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
	// self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

	nn.weightsInputHidden = mat.NewDense(nn.Config.HiddenNodes, nn.Config.InputNodes, nil)
	nn.weightsHiddenOutput = mat.NewDense(nn.Config.OutputNodes, nn.Config.HiddenNodes, nil)

	weightsInputHiddenData := nn.weightsInputHidden.RawMatrix().Data
	weightsHiddenOutputData := nn.weightsHiddenOutput.RawMatrix().Data

	for i := range weightsInputHiddenData {
		weightsInputHiddenData[i] = normalDispath(0, math.Pow(float64(nn.Config.HiddenNodes), -0.5)*nn.Config.WeightCoef)
	}
	for i := range weightsHiddenOutputData {
		weightsHiddenOutputData[i] = normalDispath(0, math.Pow(float64(nn.Config.OutputNodes), -0.5)*nn.Config.WeightCoef)
	}

	return nn
}

// Train обучает нейронную сеть, используя обратное распространение.
func (nn *NeuralNetwork) Train(inputData, outputData *mat.Dense) error {

	// Используем обратное распространение для регулировки весов.
	if err := nn.Backpropogate(inputData, outputData); err != nil {
		return err
	}

	return nil
}

// Обратное распространение.
func (nn *NeuralNetwork) Backpropogate(inputData, TargetData *mat.Dense) error {

	//сглаженными вход­ными сигналами скрытого промежуточного слоя
	//рассчитать входящие сигналы для скрытого слоя
	hiddenIn := new(mat.Dense)
	hiddenIn.Mul(nn.weightsInputHidden, inputData)

	//активируем сигналы скрытого слоя.
	//рассчитать исходящие сигналы для скрытого слоя
	hiddenOut := new(mat.Dense)
	hiddenOut.Apply(activationFunc, hiddenIn)

	//рассчитать входящие сигналы для выходного слоя
	outputIn := new(mat.Dense)
	outputIn.Mul(nn.weightsHiddenOutput, hiddenOut)

	//рассчитать исходящие сигналы для выходного слоя
	outputOut := new(mat.Dense)
	outputOut.Apply(activationFunc, outputIn)

	//Обратное распространение

	//вычисляю ошибку сети
	outputErrors := new(mat.Dense)
	outputErrors.Sub(TargetData, outputOut)

	hiddenErrors := new(mat.Dense)
	hiddenErrors.Mul(nn.weightsHiddenOutput.T(), outputErrors)

	//ΔWⱼₖ = α · Eₖ · Oₖ(1-Oₖ) · Oⱼͭ
	//				{SigBack}
	//			{LocalDelta }
	//			{  DeltaWeightSum }
	weightsRefreshFunc := func(wJK, oJ, oK, eK *mat.Dense) {
		sigBack := new(mat.Dense)
		localDelta := new(mat.Dense)
		deltaWeightSum := new(mat.Dense)

		sigBack.Apply(sigmoidBack, oK)
		localDelta.MulElem(eK, sigBack)
		deltaWeightSum.Mul(localDelta, oJ.T())
		deltaWeightSum.Scale(nn.Config.LearningRate, deltaWeightSum)
		wJK.Add(wJK, deltaWeightSum)
	}
	// обновить весовые коэффициенты связей между скрытым и выходным слоями
	weightsRefreshFunc(nn.weightsHiddenOutput, hiddenOut, outputOut, outputErrors)
	weightsRefreshFunc(nn.weightsInputHidden, inputData, hiddenOut, hiddenErrors)

	return nil
}

// Predict делает предсказание с помощью
// обученной нейронной сети.
func (nn *NeuralNetwork) Predict(inputData *mat.Dense) (*mat.Dense, error) {

	// Проверяем, представляет ли значение NeuralNetwork
	// обученную модель.
	if nn.weightsInputHidden == nil || nn.weightsHiddenOutput == nil {
		return nil, errors.New("the supplied weight are empty")
	}

	return nn.Query(inputData)
}

func (nn NeuralNetwork) Query(inputData *mat.Dense) (*mat.Dense, error) {
	//сглаженными вход­ными сигналами скрытого промежуточного слоя
	//рассчитать входящие сигналы для скрытого слоя
	hiddenIn := new(mat.Dense)
	hiddenIn.Mul(nn.weightsInputHidden, inputData)

	//активируем сигналы скрытого слоя.
	//рассчитать исходящие сигналы для скрытого слоя
	hiddenOut := new(mat.Dense)
	hiddenOut.Apply(activationFunc, hiddenIn)

	//рассчитать входящие сигналы для выходного слоя
	outputIn := new(mat.Dense)
	outputIn.Mul(nn.weightsHiddenOutput, hiddenOut)

	//рассчитать исходящие сигналы для выходного слоя
	outputOut := new(mat.Dense)
	outputOut.Apply(activationFunc, outputIn)

	return outputOut, nil
}