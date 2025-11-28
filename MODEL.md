# Download Trained Model

## Model Information

- **Name:** Landscape Inpainting GAN (Fine-tuned)
- **Performance:** 96.7% accuracy (Easy), 95.7% (Medium), 94.5% (Hard)
- **Size:** ~149 MB
- **Format:** TensorFlow/Keras (.keras file)

## Download Link

🔗 **Click here to download model from Google Drive:[https://drive.google.com/file/d/1MF3IwvqQhG0_afcGP5Rcrg9i_sKJVDDu/view?usp=sharing]**

## How to Load

import tensorflow as tf

Load the model
model = tf.keras.models.load_model(
'inpainting_generator_finetuned_best.keras',
compile=False
)

print("✓ Model loaded! Ready to use.")


## Model Details

- **Architecture:** U-Net Generator (39M params) + PatchGAN Discriminator (2.7M params)
- **Training:** 20 initial epochs + 10 fine-tuning epochs
- **Input:** 256×256×4 (RGB image + edge map)
- **Output:** 256×256×3 (inpainted RGB image)
- **Final Gen Loss:** 10.58
- **L1 Loss:** 0.0650 (93.5% pixel accuracy on training, 96.7% on test)

## Performance by Difficulty

| Difficulty | Mask Coverage | Accuracy | Visual Quality |
|------------|---------------|----------|----------------|
| Easy | 10-20% | 96.7% | 9.5/10 |
| Medium | 25-35% | 95.7% | 9.0/10 |
| Hard | 40-50% | 94.5% | 8.0/10 |

## Citation

If you use this model, please cite:

Landscape Image Inpainting with GAN
GitHub: github.com/nivedh2004/landscape-inpainting-gan
Year: 2025
