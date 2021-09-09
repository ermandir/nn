package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/ermandir/nn/pkg/ergo_mnist"
	"github.com/ermandir/nn/pkg/ergo_nnfisher"
	"gonum.org/v1/gonum/floats"
)

var start time.Time
var lastTime time.Time

const (
	//При текущих настройках эффективноть сети равна ~92%
	//При настройках 60 000 и 10 000, эффективность ~95%
	countTrainData = 6000 //максимально можно установить 60000
	countTestData  = 1000 //максимально можно установить 10000
)

func main() {
	/*
		Создаем коммандную строку: выбор метода обучения.
		И кол-во обрабатываемых данных.

		Предлагаем также сформировать слепок цифры, взяв изображение со входа сети

		Добавить сохранение весов в файл.
	*/
	start = time.Now()
	lastTime = start

	// Создаю Набор данных для обучения сети MNIST
	// представляющих собой изображения рукописных цифр
	// и соответствующей ей цифры.

	dataSet, err := ergo_mnist.ReadTrainSet("./dataset")
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("(%v)(%v)\t Загрузил БД МНИСТ (60000 записей)...\n", time.Since(start), time.Since(lastTime))
	lastTime = time.Now()

	randSrc := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSrc)
	rO := randGen.Perm(60000)
	r1 := randGen.Perm(10000)
	randomTrainOrder := rO[0:countTrainData]
	randomTestOrder := r1[0:countTestData]

	fmt.Printf("(%v)(%v)\t Перемешал обучающие данные(%v Записей)...\n", time.Since(start), time.Since(lastTime), countTrainData)
	lastTime = time.Now()

	// Train Neural Network
	n := TrainPerOnce(dataSet, randomTrainOrder)
	//n := TrainForAll(dataSet,randomTrainOrder)

	//!!!!!!!!!!!!!!!!!!!!!
	//Тестирование сети
	//!!!!!!!!!!!!!!!!!!!!!

	testDataSet, err := ergo_mnist.ReadTestSet("./dataset")
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("(%v)(%v)\t Загрузил тестовые записи(%v Записей)...\n", time.Since(start), time.Since(lastTime), countTestData)
	lastTime = time.Now()

	testingNetwork(n, testDataSet, randomTestOrder)
}

func testingNetwork(n *ergo_nnfisher.NeuralNetwork, testDataSet *ergo_mnist.DataSet, randomTestOrder []int) {
	var errs, count float32
	var errByDigit, allByDigit [10]float32

	for i := 0; i < len(randomTestOrder); i++ {
		digitImage := testDataSet.Data[randomTestOrder[i]]

		inputsTest, _ := getScaledData(digitImage.Image, digitImage.Digit)

		predictions, err := n.Predict(inputsTest)
		if err != nil {
			log.Fatal(err)
		}

		predictDigit := floats.MaxIdx(predictions.RawMatrix().Data)
		allByDigit[digitImage.Digit]++
		if int(digitImage.Digit) != predictDigit {
			errs++
			errByDigit[digitImage.Digit]++
		}
		count++
	}
	fmt.Printf("(%v)(%v)\t Собрал и скалировал тестовые данные (%v)...\n", time.Since(start), time.Since(lastTime), countTestData)
	lastTime = time.Now()

	fmt.Printf("Кол-во элементов: %v\t Эффективность: %.2f\n", countTestData, 1-errs/count)
	fmt.Printf("Ошибок: %v\t Распределение ошибок: %v\n", errs, errByDigit)
	var str string
	for i := range allByDigit {
		percent := (allByDigit[i] - errByDigit[i]) / allByDigit[i] * 100
		str = str + fmt.Sprintf("%v-%.0f%%\t", i, percent)
	}

	fmt.Printf("Угадывание по цифрам:\n%v\n", str)
}
