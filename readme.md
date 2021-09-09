# NeuralNetwork
## Overview
NeuralNetwork - Учебный проект по мотивам книги "Создаём нейронную сеть на Python" Тарика Ращида. Проект выполнен на новом для меня языке - GoLang. Проект написан исключительно в учебных целях.

Проект содержит два проекта - ergo_nnfisher и ergo_mnist

## ergo_nnfisher

Включает основную логику нейронной сети из трёх слоёв. Использует в качестве активационной ф-ции сигмоиду.
* Тренировку
* Предсказание

## ergo_mnist

Пакет для подготовки данных из [`THE MNIST DATABASE`](http://yann.lecun.com/exdb/mnist/).

Пакет сделан на основе чужого кода, не помню уже откуда взятого. Был взят пакет и приведен в рабочее состояние.

### `ergo_mnist.ReadTrainSet("./dataset")`
С помощью этой функции пакет "натравливается" на папку с разархивированными данными с сайта. Имена файлов должны быть:

```go
 	TrainImagesFile = "train-images.idx3-ubyte"
	TrainLabelsFile = "train-labels.idx1-ubyte"
	TestImagesFile  = "t10k-images.idx3-ubyte"
	TestLabelsFile  = "t10k-labels.idx1-ubyte"
```

## Запуск проекта
Запуск проекта производится из папки  **md/ergo_nnfisher_mnist** именно там располагать папку **datset** в которой и будут находиться файлы MNIST.
