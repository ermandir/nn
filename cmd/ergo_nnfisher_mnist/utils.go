package main

import (
	"fmt"
	"time"

	"github.com/ermandir/nn/pkg/ergo_mnist"
	"github.com/ermandir/nn/pkg/ergo_nnfisher"
	"gonum.org/v1/gonum/mat"
)

func TrainPerOnce(dataSet *ergo_mnist.DataSet, randomTrainOrder []int) (n *ergo_nnfisher.NeuralNetwork) {
	settings := ergo_nnfisher.NeuralNetConfig{
		InputNodes:   784,
		HiddenNodes:  200,
		OutputNodes:  10,
		NumEpochs:    2,
		LearningRate: 0.1,
		WeightCoef: 1.0,
	}

	n = ergo_nnfisher.NewNetwork(settings)

	fmt.Printf("Обучаю сеть по одной записи!!! всего проходов: %v проход: ", settings.NumEpochs*len(randomTrainOrder))
	count := settings.NumEpochs*len(randomTrainOrder)/100

	for i := 0; i < n.Config.NumEpochs; i++ {
		for j := 0; j < len(randomTrainOrder); j++ {
			digitImage := dataSet.Data[randomTrainOrder[j]]

			//Тренируем сеть
			n.Train(getScaledData(digitImage.Image, digitImage.Digit))

			if (j)%100 == 0 {
				fmt.Printf(" %v",count)
				count--
			}
		}
	}
	
	fmt.Printf("\n(%v)(%v)\t Тренировка сети окончена...\n", time.Since(start), time.Since(lastTime))
	lastTime = time.Now()

	return
}

func TrainForAll(dataSet *ergo_mnist.DataSet, randomTrainOrder []int) (n *ergo_nnfisher.NeuralNetwork) {

	settings := ergo_nnfisher.NeuralNetConfig{
		InputNodes:   784,
		HiddenNodes:  200,
		OutputNodes:  10,
		NumEpochs:    50,
		LearningRate: 0.00018,
		WeightCoef: 1.0,
	}

	n = ergo_nnfisher.NewNetwork(settings)

	var imagesData []float64
	var digitsData []float64

	for i := 0; i < len(randomTrainOrder); i++ {
		digitImage := dataSet.Data[randomTrainOrder[i]]
		img := getScaledImage(digitImage.Image)
		dgt := getAnswerFromDigit(digitImage.Digit)

		imagesData = append(imagesData, img...)
		digitsData = append(digitsData, dgt...)
	}

	fmt.Printf("(%v)(%v)\t Собрал и скалировал входные данные (%v)...\n", time.Since(start), time.Since(lastTime), countTrainData)
	lastTime = time.Now()

	inputs := mat.NewDense(countTrainData, dataSet.Height*dataSet.Width, imagesData)
	targets := mat.NewDense(countTrainData, n.Config.OutputNodes, digitsData)

	fmt.Printf("(%v)(%v)\t Создал матрицы...\n", time.Since(start), time.Since(lastTime))
	lastTime = time.Now()

	inputsT := mat.DenseCopyOf(inputs.T())
	targetsT := mat.DenseCopyOf(targets.T())

	fmt.Printf("(%v)(%v)\t Транспонировал матрицы...\n", time.Since(start), time.Since(lastTime))
	lastTime = time.Now()

	//Тренируем сеть
	fmt.Printf("Обучаю сеть!!! всего проходов: %v проход: ", settings.NumEpochs)

	for i := 0; i < n.Config.NumEpochs; i++ {
		n.Train(inputsT, targetsT)
		fmt.Printf("%v", i+1)
	}
	fmt.Printf("\n(%v)(%v)\t Обучение сети окончено...\n", time.Since(start), time.Since(lastTime))
	lastTime = time.Now()

	return
}

func getScaledData(image [][]uint8, digit uint8) (imageDense, digitDense *mat.Dense) {
	imageData := getScaledImage(image)
	digitData := getAnswerFromDigit(digit)

	return mat.NewDense(784, 1, imageData), mat.NewDense(10, 1, digitData)
}

//подготовка данных MNIST для InputArray
/*
	Числа в диапазоне 0-255 надо превратить в
	числа в диапазоне 0,01 - 1.0
	y = (( x / 255 ) * 0.99) + 0.01
*/
//ScaledInput = (dataSet.Data[0].Image[y,x])
//получаю скалированный набор данных с индексом index
// получаю скалированный набор данных с индексом index
// так как изображение rowfirst, то переделываем на columnfirst
func getScaledImage(image [][]uint8) []float64 {
	lenghtH := len(image)
	if lenghtH != 28 {
		panic("Count Rows not equal 28")
	}
	lenghtW := len(image[0])
	if lenghtH != 28 {
		panic("Count Cols not equal 28")
	}

	out := make([]float64, lenghtH*lenghtW)
	var index int
	for i := 0; i < lenghtH; i++ {
		for j := 0; j < lenghtW; j++ {
			out[index] = ((float64(image[i][j]) / 255.0) * 0.99) + 0.01
			//log.Println(i*(lenghtH)+j," ", index)
			index++
		}
	}
	return out
}

func getAnswerFromDigit(digit uint8) []float64 {
	answer := []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}
	answer[digit] = 0.99
	return answer
}