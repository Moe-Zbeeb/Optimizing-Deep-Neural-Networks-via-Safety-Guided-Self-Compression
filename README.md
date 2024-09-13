# Safety-Driven Self-Compressing Neural Networks

This project explores a **simple and general neural network weight compression approach**. The core concept is to represent the network parameters (weights and biases) in a **latent space**, effectively reparameterizing the model. The ultimate goal is to significantly reduce the model's size while preserving its performance and generalizability.
---

**Note:** This README serves as a living document to track the development of ideas and progress throughout the project.

---

## Choosing the Safety Set: Algorithm Ensembling
[Safet set is living here](https://mailaub-my.sharepoint.com/:f:/r/personal/mbz02_mail_aub_edu/Documents/safet_set?csf=1&web=1&e=UftuGF)
The process of selecting the **safety set** involved an ensemble of three distinct algorithms. This approach ensures that the selected images from the **CIFAR-10 dataset** are **diverse**, **challenging**, and **representative**. Here's a breakdown of the methodology:

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

By combining these three techniques, we obtained a safety set that includes **diverse**, **challenging**, and **critical** examples. This safety set is designed to ensure that the compressed model retains its performance across a wide range of scenarios from the **CIFAR-10 dataset**.

---

