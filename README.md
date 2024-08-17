# Uncertainty-Guided Context Consistency Learning for Semi-supervised Semantic Segmentation
Abstract:— Semi-supervised semantic segmentation has attracted considerable attention for its ability to mitigate the reliance on extensive labeled data. However, existing consistency regularization methods only utilize high certain pixels with prediction confidence surpassing a fixed threshold for training, failing to fully leverage the potential supervisory information within the network.  Therefore, this paper proposes the uncertainty-participation  context consistency learning (\textbf{UCCL}) method to explore richer supervisory signals. Specifically, we first design the semantic backpropagation update (SBU) strategy to fully exploit the knowledge from uncertain pixel regions, enabling the model to learn consistent pixel-level semantic information from those areas. Furthermore, we propose the class-aware knowledge regulation (CKR) module to facilitate the regulation of class-level semantic features across different augmented views, promoting consistent learning of class-level semantic information within the encoder.
Experimental results on two public benchmarks demonstrate that our proposed method achieves state-of-the-art performance. 




