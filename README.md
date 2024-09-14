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
The overall loss function is defined as:

$$
\Lambda(x) = \Lambda_0(x) + \gamma \cdot Q + \lambda \cdot \text{safety\_loss}
$$

Where:

- **$\Lambda_0(x)$**: The original loss (e.g., cross-entropy loss).
- **$\gamma$**: A regularization parameter that controls the trade-off between accuracy and compression.
- **$Q$**: Quantization loss, which is based on the number of bits used for weights.
- **$\lambda$**: A regularization term that balances the safety loss.
- **$\text{safety\_loss}$**: A term that measures the performance drop on a preselected "safety set" (critical examples). This ensures the network does not degrade in accuracy on important data points when compressed.

The **safety loss** helps to maintain accuracy on the most critical examples, particularly when aggressive compression is applied.
