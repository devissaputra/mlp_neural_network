from pathlib import Path
import json, matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,f1_score,ConfusionMatrixDisplay
X,y=load_digits(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y); clf=MLPClassifier(hidden_layer_sizes=(128,64),max_iter=450,early_stopping=True,random_state=42); m=Pipeline([('s',StandardScaler()),('m',clf)]); m.fit(Xtr,ytr); pred=m.predict(Xte); out={'accuracy':float(accuracy_score(yte,pred)),'macro_f1':float(f1_score(yte,pred,average='macro')),'epochs':int(len(clf.loss_curve_))}
Path('results').mkdir(exist_ok=True); Path('results/metrics.json').write_text(json.dumps(out,indent=2))
plt.figure(figsize=(7,5)); plt.plot(clf.loss_curve_); plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.title('MLP optimization'); plt.tight_layout(); plt.savefig('assets/03_data_or_model.png',dpi=150); plt.close()
fig,ax=plt.subplots(figsize=(7,6)); ConfusionMatrixDisplay.from_predictions(yte,pred,ax=ax,colorbar=False); ax.set_title('Held-out confusion matrix'); fig.tight_layout(); fig.savefig('assets/04_evaluation_or_results.png',dpi=150); plt.close(fig); print(json.dumps(out,indent=2))