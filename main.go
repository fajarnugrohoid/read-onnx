package main

import (
	"fmt"
	"log"

	"github.com/yalue/onnxruntime_go"
)

func main() {
	fmt.Println("ONNX Runtime Version:", onnxruntime_go.GetVersion())

	if err := onnxruntime_go.InitializeEnvironment(); err != nil {
		log.Fatalf("Failed to initialize ONNX Runtime: %v", err)
	}
	defer onnxruntime_go.DestroyEnvironment()

	modelPath := "knn_model_fixed.onnx"

	// --- Input Data ---
	inputData := []float32{97, 91, 58, 15}
	inputShape := []int64{1, 4}
	inputTensor, err := onnxruntime_go.NewTensor(inputShape, inputData)
	if err != nil {
		log.Fatalf("Failed to create input tensor: %v", err)
	}

	// --- Output Tensor ---
	outputLabel, err := onnxruntime_go.NewEmptyTensor[int64]([]int64{1})
	if err != nil {
		log.Fatalf("Failed to create output tensor: %v", err)
	}

	// --- Create DynamicSession ---
	session, err := onnxruntime_go.NewDynamicSession[float32, int64](
		modelPath,
		[]string{"float_input"},
		[]string{"label"},
	)
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}
	defer session.Destroy()

	// --- Run Inference ---
	err = session.Run(
		[]*onnxruntime_go.Tensor[float32]{inputTensor},
		[]*onnxruntime_go.Tensor[int64]{outputLabel},
	)
	if err != nil {
		log.Fatalf("Failed to run inference: %v", err)
	}

	// --- Get Output ---
	labels := outputLabel.GetData()
	if err != nil {
		log.Fatalf("Failed to get label output: %v", err)
	}
	fmt.Println("Predicted Label:", labels)
}
