\subsection{Support Vector Machine Performance}

We assessed the SVM classifier using three validation strategies: a train-test split, 10-fold stratified cross-validation, and leave-one-out cross-validation (LOOCV). The RBF kernel consistently outperformed linear, polynomial, and sigmoid kernels in repeated stratified cross-validation and was selected for final model development.

\begin{table}[H]
\centering
\resizebox{\linewidth}{!}{%
\begin{tabular}{p{3.5cm}cccc}
\toprule
\textbf{Method} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1 Score} \
\midrule
\textbf{Train/Test Split} & \textbf{0.8804} & \textbf{0.8846} & \textbf{0.9118} & \textbf{0.8979} \
Stratified K-Fold & $\sim$0.87 & $\sim$0.88 & $\sim$0.89 & $\sim$0.88 \
LOOCV & $\sim$0.87 & $\sim$0.88 & $\sim$0.89 & $\sim$0.88 \
\bottomrule
\end{tabular}%
}
\caption{Support Vector Machine performance across validation methods.}
\end{table}

The grid search tuning of 
𝐶
C and 
𝛾
γ parameters yielded the best F1 score for 
𝐶
=
10
C=10 and 
𝛾
=
0.1
γ=0.1 using the RBF kernel. On the test set, the tuned model achieved an ROC AUC score of approximately 0.927, indicating excellent discriminative ability.

The confusion matrix showed that the model was especially effective in identifying heart disease cases, with high recall and precision for the positive class. Overall, the SVM provided competitive and reliable performance, comparable to the random forest and neural network models, while offering a robust margin-based decision boundary.