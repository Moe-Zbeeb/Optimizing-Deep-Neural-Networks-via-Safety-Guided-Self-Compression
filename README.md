# Safety-Driven Self-Compressing Neural Networks

This project explores a **simple and general neural network weight compression approach**. The core concept is to represent the network parameters (weights and biases) in a **latent space**, effectively reparameterizing the model. The ultimate goal is to significantly reduce the model's size while preserving its performance and generalizability.
---

**Note:** This README serves as a living document to track the development of ideas and progress throughout the project.

---

## Choosing the Safety Set: Algorithm Ensembling
[Safet Set living here cifar10](https://mailaub-my.sharepoint.com/:f:/r/personal/mbz02_mail_aub_edu/Documents/safet_set?csf=1&web=1&e=UftuGF)   
[Safet Set living here MNIST](https://mailaub-my.sharepoint.com/:f:/g/personal/mbz02_mail_aub_edu/EtfJiH17wwtOrmgitadVf1QBB-0wIbUtNIBCXirfJp9RSQ?e=KOnePE
)

The process of selecting the **safety set** involved an ensemble of three distinct algorithms. This approach ensures that the selected images from the **MNIST dataset** are **diverse**, **challenging**, and **representative**. Here's a breakdown of the methodology: 

---  

**Note:** This is an inference Mode Algorithm

---
### 1. **Grad-CAM (Gradient-weighted Class Activation Mapping)**
   - We applied **Grad-CAM** to rank images based on the importance of the input regions that the model relied on for its predictions.
   - The images selected were those where the model's prediction was influenced by **critical features**, i.e., regions of the image that played a significant role in the decision-making process.
   
### 2. **Uncertainty Sampling**
   - To capture the most **difficult** images, we used an uncertainty strategy, selecting examples where the model was unsure of its predictions.
   - This was measured by identifying images where the difference between the two most probable classes in the **softmax output** was minimal, indicating that the model struggled to classify these images.

### 3. **Clustering for Diversity**
   - To ensure that the safety set was **representative of the full feature space**, we applied a clustering algorithm.
   - We projected the embeddings from the **last convolutional layer** of the model into a lower-dimensional space and performed **K-Means clustering**. From each cluster, we selected an equal number of images, ensuring that the safety set is diverse and representative of various patterns in the dataset.

---

By combining these three techniques, we obtained a safety set that includes **diverse**, **challenging**, and **critical** examples. This safety set is designed to ensure that the compressed model retains its performance across a wide range of scenarios from the **MNIST dataset**.

---
## Quantization Functon  

The quantization function is defined as:

$$
q(x, b, e) = 2^{e} \left\lfloor \min\left(\max\left(2^{-e} \cdot x, -2^{b-1}\right), 2^{b-1} - 1 \right)\right\rfloor
$$

Where:

- **x**: Input data
- **b**: Bit depth used for quantization (learnable) 
- **e**: Exponent used for scaling (learnable)
---  
## Loss Function  
The overall loss function is given by: 

$$
Λ(x) = Λ₀(x) + γ * Q + λ * safety_loss  
$$

Where:

- **Λ₀(x)**: The original loss function (e.g., cross-entropy loss).
- **γ**: A regularization parameter for controlling the compression-accuracy tradeoff.
- **Q**: Avergae size in bytes of neural net 
- **λ**: A regularization parameter controlling the importance of the safety set.
- **safety_loss**: A penalty term for evaluating model performance on a predefined safety set.
---  
## Training Loop  
## Safe Quantization Training Loop Pseudocode

1. **Initialize Variables:**
   - Set `prev_safety_acc` to `None`
   - Set `safety_acc_drop_threshold` to a predefined value (e.g., 5.0 for 5% drop in accuracy)
   - Initialize lists to track `test_accs`, `bytes_used`, and `safety_losses`
   - Calculate `initial_safety_acc` as the starting accuracy on the safety set.
   - Assign `prev_safety_acc` to `initial_safety_acc`.

2. **Training Loop (4000 iterations):**

   ```python
   for i in range(4000):
       # Step 1: Perform a training step
       loss, Q, safety_loss = train_step()

       # Step 2: Calculate model size in bytes based on quantized bits
       model_bytes = Q / 8 * weight_count

       # Step 3: Every 10 iterations:
       if i % 10 == 9:
           # Step 3.1: Calculate test accuracy
           test_acc = get_test_acc()

           # Step 3.2: Calculate accuracy on safety set
           safety_acc = get_safety_acc()

           # Step 3.3: Compute accuracy drop from the previous safety evaluation
           acc_drop = prev_safety_acc - safety_acc

           # Step 3.4: If the drop exceeds the threshold
           if acc_drop > safety_acc_drop_threshold:
               # Step 3.5: Check for kernels with zero bits
               if check_zero_bit_kernels():
                   # Step 3.6: Restore kernels with zero bits (e.g., restore 50% of them)
                   restore_zero_bit_kernels(restore_fraction=0.5)

           # Step 3.7: Update the previous safety accuracy to the current one
           prev_safety_acc = safety_acc

       # Else: Use the previous test accuracy
       else:
           test_acc = test_accs[-1] if test_accs else 0.0

       # Step 4: Log test accuracy, model size, and safety loss
       test_accs.append(test_acc)
       bytes_used.append(model_bytes)
       safety_losses.append(safety_loss)
---


