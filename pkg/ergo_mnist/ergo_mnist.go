package ergo_mnist

import (
	"fmt"
	"os"
	"path/filepath"
)

const (
	//Здесь нужно указать названия файлов, взятых из
	//mnist.org
	TrainImagesFile = "train-images.idx3-ubyte"
	TrainLabelsFile = "train-labels.idx1-ubyte"
	TestImagesFile  = "t10k-images.idx3-ubyte"
	TestLabelsFile  = "t10k-labels.idx1-ubyte"

	labelFileMagic = 0x00000801
	imageFileMagic = 0x00000803

	msgInvalidFormat = "invalid format: %s"
	msgSizeUnmatch   = "data size does not match: %s %s"
)

func fileError(f *os.File) error {
	return fmt.Errorf(msgInvalidFormat, f.Name())
}

//read 4 bytes and convert to MSB first integer
func readInt32(f *os.File) (int, error) {
	buf := make([]byte, 4)
	n, e := f.Read(buf)
	switch {
	case e != nil:
		return 0, e
	case n != 4:
		return 0, fileError(f)
	}
	v := 0
	for _, x := range buf {
		v = v<<8 + int(x)
	}

	return v, nil
}

//collection image data
type imageData struct {
	Count  int
	Width  int
	Height int
	Data   []uint8
}

//read a formatted file
/*
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
*/
func readImagesFile(path string) (*imageData, error) {
	f, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	defer f.Close()

	magic, e := readInt32(f)
	if e != nil || magic != imageFileMagic {
		return nil, fileError(f)
	}

	count, e := readInt32(f)
	if e != nil || magic != imageFileMagic {
		return nil, fileError(f)
	}

	height, e := readInt32(f)
	if e != nil || magic != imageFileMagic {
		return nil, fileError(f)
	}

	width, e := readInt32(f)
	if e != nil || magic != imageFileMagic {
		return nil, fileError(f)
	}

	size := count * height * width
	data := &imageData{count, width, height, make([]uint8, size)}
	len, err := f.Read(data.Data)
	if err != nil || len != size {
		return nil, fileError(f)
	}
	return data, nil
}

//collection label data
type labelData struct {
	Count int
	Data  []uint8
}

func readLabelsFile(path string) (*labelData, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	magic, err := readInt32(f)
	if err != nil || magic != labelFileMagic {
		return nil, fileError(f)
	}

	count, err := readInt32(f)
	if err != nil {
		return nil, fileError(f)
	}

	data := &labelData{count, make([]uint8, count)}
	len, err := f.Read(data.Data)
	if err != nil || len != count {
		return nil, err
	}

	return data, nil
}

//single image+digit data
type DigitImage struct {
	Digit uint8
	Image [][]uint8
}

type DataSet struct {
	Count  int
	Width  int
	Height int
	Data   []DigitImage
}

func ReadDataSet(imagesPath, labelsPath string) (*DataSet, error) {
	images, err := readImagesFile(imagesPath)
	if err != nil {
		return nil, err
	}
	labels, err := readLabelsFile(labelsPath)
	if err != nil {
		return nil, err
	}

	if images.Count != labels.Count {
		return nil, fmt.Errorf(msgSizeUnmatch, labelsPath, imagesPath)
	}

	dataSet := &DataSet{
		Count:  images.Count,
		Width:  images.Width,
		Height: images.Height,
	}
	dataSet.Data = make([]DigitImage, dataSet.Count)
	rows := splitToRows(images.Data, images.Count, images.Height)
	for i := 0; i < dataSet.Count; i++ {
		data := &dataSet.Data[i]
		data.Digit = labels.Data[i]
		data.Image = rows[0:dataSet.Width]
		rows = rows[dataSet.Width:]
	}
	return dataSet, nil
}

func ReadTrainSet(dir string) (*DataSet, error) {
	imagesPath := filepath.Join(dir, TrainImagesFile)
	labelsPath := filepath.Join(dir, TrainLabelsFile)
	return ReadDataSet(imagesPath, labelsPath)
}

func ReadTestSet(dir string) (*DataSet, error) {
	imagesPath := filepath.Join(dir, TestImagesFile)
	labelsPath := filepath.Join(dir, TestLabelsFile)
	return ReadDataSet(imagesPath, labelsPath)
}

func splitToRows(data []uint8, Count, Height int) [][]uint8 {
	countRows := Count * Height
	rows := make([][]uint8, countRows)
	for i := 0; i < countRows; i++ {
		rows[i] = data[0:Height]
		data = data[Height:]
	}
	return rows
}
