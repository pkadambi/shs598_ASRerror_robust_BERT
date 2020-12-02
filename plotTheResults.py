import matplotlib.pyplot as plt
import numpy as np

#from the results csv files, pasted here
WER_LEVELS = np.array([5, 10, 15, 20, 25, 30])
baseline = [.443, .384, .335, .25, .247, .194]
t1=[.397, 0.392, .345, .28, .23, .209]
t1point5 = [.444, .397, .356, .25, .183, .237]
t2= [.418, .403, .314, .319, .213, .245]
t3=[0.436, .401, .292, .251, .207, .16]
t4=[0.45, .4, .27, .292, .232, .236]
t5 = [1,1,1,1,1]

plt.plot([0,30],[.57, 0.57],'--k', label='No ASR Noise')
plt.plot([0, 5, 10, 15, 20, 25, 30], [.57, .443, .384, .335, .25, .247, .194], '-bo', label='No Regularization')
plt.legend()
plt.grid(True)
plt.ylabel('Matthews Corr. (CoLA dev set)')
plt.xlabel('ASR Word Error Rate (%)')
plt.title('Effect of ASR Errors on BERT LM (CoLA)')

plt.figure()
plt.plot(WER_LEVELS, baseline, '-bo', label='No Regularization')
plt.plot(WER_LEVELS, t1, '-rx', label='T=1')
plt.plot(WER_LEVELS, t1point5, '-gx', label='T=1.5')
plt.plot(WER_LEVELS, t2, '-kx', label='T=2')
plt.plot(WER_LEVELS, t3, '-mx', label='T=3')
plt.plot(WER_LEVELS, t4, '-yx', label='T=4')
plt.legend()
plt.grid(True)
plt.ylabel('Matthews Corr. (CoLA dev set)')
plt.xlabel('ASR Word Error Rate  (%)')
plt.title('Distillation for BERT Robustness to ASR Errors')
# plt.title('WER vs MCC, Effect of Temp.')
plt.show()


plt.figure()
plt.plot([0,10],[58.4, 58.4],'--k', label='No ASR Noise')
plt.plot([0, 10], [57.5, ], '-bo', label='No Regularization')
plt.legend()
plt.grid(True)
plt.ylabel('Acc. (CoLA dev set)')
plt.xlabel('ASR Word Error Rate (%)')
plt.title('Effect of ASR Errors on BERT (FluentAI Intent Recognition)')
