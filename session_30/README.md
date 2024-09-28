# Multimodal Phi-2 Fine-tuning

This project fine-tunes the Microsoft Phi-2 language model to create a multimodal AI system capable of processing both text and image inputs.

## Project Overview

This project aims to enhance the Phi-2 language model with image understanding capabilities by integrating CLIP embeddings and using a custom projection layer. The result is a multimodal AI system that can process both text and image inputs, making it suitable for tasks such as image captioning, visual question answering, and more.

## Demo
The app is deployed on Hugging Face Spaces [here](https://huggingface.co/spaces/dhairyashil/multimodal_phi2).

## Key Components

1. **Phi-2 Model**: The base language model from Microsoft.
2. **CLIP**: Used for generating image embeddings.
3. **Custom Projection Layer**: Bridges the gap between CLIP embeddings and Phi-2's input space.
4. **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning technique applied to specific layers of Phi-2.

## Dataset
The images were obtained from [COCO-18GB-train2017.zip](http://images.cocodataset.org/zips/train2017.zip) and corrsponding conversations were obtained from [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K).


## Methodology

1. **Data Preparation**: 
   - Load and preprocess the instruction dataset.
   - Generate CLIP embeddings for associated images.

2. **Model Architecture**:
   - Load Phi-2 model with 4-bit quantization.
   - Implement a custom projection layer to map CLIP embeddings to Phi-2's input space.
   - Create a MultimodalPhiWithAdapter class that combines Phi-2 with the projection layer.

3. **Training**:
   - Apply LoRA for efficient fine-tuning.
   - Use a custom dataset class (ModifiedInstructDataset) for batching.
   - Implement early stopping based on loss convergence.

4. **Optimization**:
   - Use mixed-precision training (fp16).
   - Implement gradient checkpointing for memory efficiency.

## Key Features

- **Multimodal Capability**: Processes both text and image inputs.
- **Memory Efficient**: Uses 4-bit quantization and LoRA for reduced memory footprint.
- **Customizable**: Easy to adjust hyperparameters and model architecture.

## Limitations and Considerations

1. **Hardware Requirements**: Requires a GPU with sufficient VRAM (tested on Tesla P100-PCIE-16GB).
2. **Dataset Dependency**: Performance heavily relies on the quality and diversity of the instruction dataset.
3. **Fine-tuning Scope**: Only adapter layers and projection layer are fine-tuned, which may limit the model's adaptability.
4. **Quantization Effects**: 4-bit quantization may impact model performance compared to full-precision training.

## Usage

1. Ensure all required libraries are installed (transformers, peft, accelerate, etc.).
2. Prepare your instruction dataset and image embeddings.
3. Adjust hyperparameters in the training arguments as needed.
4. Run the notebook to train the model.
5. Use the fine-tuned model for inference on new text-image inputs.

## Future Work

1. Experiment with different projection layer architectures.
2. Try fine-tuning more layers of the base Phi-2 model.
3. Evaluate on diverse multimodal datasets to assess generalization.
4. Implement inference pipeline for easy use of the fine-tuned model.

## Acknowledgements

- Microsoft for the Phi-2 model
- OpenAI for the CLIP model
- Hugging Face for the transformers library
- PEFT library contributors


