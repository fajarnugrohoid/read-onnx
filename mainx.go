package main

import (
	"fmt"
	"log"

	"github.com/yalue/onnxruntime_go"
)

func main() {

	fmt.Println("ONNX Runtime Version:", onnxruntime_go.GetVersion())

	err := onnxruntime_go.InitializeEnvironment()
	if err != nil {
		log.Fatalf("Error initializing ONNX Runtime: %v", err)
	}
	defer onnxruntime_go.DestroyEnvironment()

	// Path to your ONNX model
	modelPath := "knn_model_fixed.onnx"

	// Prepare input data
	inputData := []float32{1, 2, 3, 4} // adjust based on model
	inputShape := []int64{1, 4}        // adjust shape accordingly

	// Create input tensor
	inputTensor, err := onnxruntime_go.NewTensor(inputShape, inputData)
	if err != nil {
		log.Fatalf("Failed to create input tensor: %v", err)
	}

	// Placeholder for output (you can also prepare an empty tensor)
	outputLabel, err := onnxruntime_go.NewEmptyTensor[float32]([]int64{1, 2}) // adjust output shape
	if err != nil {
		log.Fatalf("Failed to create output tensor: %v", err)
	}

	// Input/output names as defined in your ONNX model
	inputNames := []string{"float_input"} // check your actual input name
	outputNames := []string{"label"}      // check your actual output name

	inputs := []*onnxruntime_go.Tensor[float32]{inputTensor}
	outputs := []*onnxruntime_go.Tensor[float32]{outputLabel}
	// Create session
	session, err := onnxruntime_go.NewSession(
		modelPath,
		inputNames,
		outputNames,
		inputs,
		outputs,
	)
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}
	defer session.Destroy()

	// Run inference
	err = session.Run()
	if err != nil {
		log.Fatalf("Error running inference: %v", err)
	}

	// Print the output
	fmt.Println("Model Output:", outputLabel)
}
