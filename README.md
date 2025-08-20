---

### Author & Original Work

This work was carried out under the supervision of  **Prof. Wen-Nung Lie**  
[Google Scholar Profile](https://scholar.google.com.my/citations?user=Lv6q7ioAAAAJ&hl=en)  
Department of Electrical Engineering,  
National Chung Cheng University, Taiwan

This project is an improved version of the original work by **Phung Huu Tai**, M.Sc. student at **National Chung Cheng University, Taiwan**.

The improvements presented here were developed by **Nguyen Quang Khai**, building upon the original framework.

---

### Proposed Improvement

To enhance the accuracy of head pose estimation, I implemented a key modification in the data sampling strategy during model training:

#### Gaussian Sampling vs. Random Sampling

- **Problem**: Random sampling often introduces noise and uneven distribution across the pose space, resulting in poor generalization—especially at extreme yaw/pitch angles.
- **Solution**: Replaced random sampling with **Gaussian distribution sampling**, concentrating more samples around common pose ranges while still covering edge cases.
- **Result**:
  - **Significant RMSE improvement** (up to ~20% reduction) over random sampling.
  - Model learns a more structured representation of head pose, resulting in better generalization to real-world data.

#### Lightweight Model Retained

- Model size remains only **~0.66MB**.
- No complex or deep backbone used.
- Optimized for edge deployment and real-time applications.

---

For training logs, benchmarks, or further technical details, please contact:  
📩 **[nguyenquangk981@gmail.com](mailto:nguyenquangk981@gmail.com)**

---

### 📎 Resources

- [Download PDF Report](documents/Internship_Khai_Slide.pdf)  
- [Download Final Presentation Slides](documents/Final_Report.pdf)
