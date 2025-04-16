package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"sort"

	"github.com/yalue/onnxruntime_go"
)

func main() {
	fmt.Println("ONNX Runtime Version:", onnxruntime_go.GetVersion())

	if err := onnxruntime_go.InitializeEnvironment(); err != nil {
		log.Fatalf("Failed to initialize ONNX Runtime: %v", err)
	}
	defer onnxruntime_go.DestroyEnvironment()

	modelPath := "school_recommender_mlp.onnx"

	// --- Input: original student data
	rawInput := []float32{97, 91, 58, 107.76574571313512, -6.956058741464109, 443.45}

	// --- Weights (sqrt([0.2, 0.2, 0.2, 0.4]))
	weights := []float32{
		float32(math.Sqrt(0.2)),
		float32(math.Sqrt(0.2)),
		float32(math.Sqrt(0.2)),
		float32(math.Sqrt(0.2)),
		float32(math.Sqrt(0.2)),
		float32(math.Sqrt(0.4)),
	}

	// --- Input Data ---
	// --- Apply element-wise multiplication
	inputData := make([]float32, len(rawInput))
	for i := range rawInput {
		inputData[i] = rawInput[i] * weights[i]
	}

	inputShape := []int64{1, int64(len(inputData))}
	//inputData := []float32{97, 91, 58, 443.45}
	//inputShape := []int64{1, 4}
	inputTensor, err := onnxruntime_go.NewTensor(inputShape, inputData)
	if err != nil {
		log.Fatalf("Failed to create input tensor: %v", err)
	}

	// --- Output Tensor ---
	var labelEncoder []int
	labelFile, err := os.Open("label_encoder.json")
	if err != nil {
		log.Fatal("Failed to open label encoder:", err)
	}
	defer labelFile.Close()
	json.NewDecoder(labelFile).Decode(&labelEncoder)

	//outputLabel, err := onnxruntime_go.NewEmptyTensor[int64]([]int64{1, int64(len(labelEncoder))})
	outputLabel, err := onnxruntime_go.NewEmptyTensor[float32]([]int64{1, int64(len(labelEncoder))})

	if err != nil {
		log.Fatalf("Failed to create output tensor: %v", err)
	}

	// --- Create DynamicSession ---
	session, err := onnxruntime_go.NewDynamicSession[float32, float32](
		modelPath,
		[]string{"float_input"},
		[]string{"probabilities"},
	)
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}
	defer session.Destroy()

	// --- Run Inference ---
	err = session.Run(
		[]*onnxruntime_go.Tensor[float32]{inputTensor},
		[]*onnxruntime_go.Tensor[float32]{outputLabel},
	)
	if err != nil {
		log.Fatalf("Failed to run inference: %v", err)
	}

	// --- Get Output ---
	probs := outputLabel.GetData()
	if err != nil {
		log.Fatalf("Failed to get label output: %v", err)
	}
	fmt.Println("Predicted probs:", probs)

	// --- Get Top 3 Indices ---
	type indexProb struct {
		Index int
		Prob  float32
	}

	var indexedProbs []indexProb
	for i, p := range probs {
		indexedProbs = append(indexedProbs, indexProb{i, p})
	}

	sort.Slice(indexedProbs, func(i, j int) bool {
		return indexedProbs[i].Prob > indexedProbs[j].Prob
	})

	// --- Print Top 3 Schools ---
	fmt.Println("Top 3 Recommended Schools:")
	for i := 0; i < 5 && i < len(indexedProbs); i++ {
		idx := indexedProbs[i].Index
		fmt.Printf("%d. %d (Prob: %.2f)\n", i+1, labelEncoder[idx], probs[idx])
	}

}
