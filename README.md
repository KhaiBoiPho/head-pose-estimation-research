# Head Pose Estimation – Gaussian Sampling Improvement

---

### Author & Original Work

This work was carried out under the supervision of **Prof. Wen-Nung Lie**  
[Google Scholar Profile](https://scholar.google.com.my/citations?user=Lv6q7ioAAAAJ&hl=en)  
Department of Electrical Engineering,  
National Chung Cheng University, Taiwan  

This project is an improved version of the original work by **Phung Huu Tai**, M.Sc. student at **National Chung Cheng University, Taiwan**.  

The improvements presented here were developed by **Nguyen Quang Khai**, building upon the original framework.  

---

### Proposed Improvement

To enhance the accuracy of head pose estimation, I focused on improving the **reference image sampling strategy** during model training.  

#### Gaussian Sampling vs. Random Sampling  

**Problem**  
- Random sampling of reference images often leads to noisy and uneven distribution across the pose space.  
- This causes weak generalization, especially at extreme yaw and pitch angles.  

**Solution**  
- Introduced **Gaussian-based sampling** instead of uniform random selection.  
- Sampling is centered around the yaw angle of the input image.  
- A smaller number of samples are drawn from opposing angles to preserve diversity.  

**Findings**  
- Clear accuracy improvement under both **Protocol 1** and **Protocol 2**.  
- Best results achieved with **σ = 30**.  
- Combining **similar + contrasting yaw angles** yields better generalization than using only the closest angles.  

**Conclusion**  
- A balanced **Gaussian sampling strategy** ensures the model learns both:  
  - **Proximity** → captures important similarities.  
  - **Diversity** → incorporates contrasting orientations.  
- This significantly improves fine-grained head pose estimation performance.  

---

### Lightweight Model Retained  

- Model size remains only **~0.66MB**.  
- No deep or heavy backbone introduced.  
- Fully optimized for **real-time, edge deployment** scenarios (e.g., AR/VR, driver monitoring).  

---

### Contact & Notes  

For training logs, benchmarks, or further technical details, please contact:  
[nguyenquangk981@gmail.com](mailto:nguyenquangk981@gmail.com)  

**Important Note**  
- This research is a **supportive extension** of the original author’s work and is **not yet published as a paper**.  
- To protect the confidentiality of the original architecture, I have only included the parts that I directly contributed and improved.  
- If you have questions, or wish to understand the **original framework in more detail**, please contact me directly via email.  

---

### Resources  

- [Download PDF Report](documents/Internship_Khai_Slide_version.pdf)  
- [Download Final Presentation Slides](documents/Final_Report_version.pdf)  

---
### TEEP scholarship

![TEEP](documents/2_CCU.png)
