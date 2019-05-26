import numpy as np
import matplotlib.pyplot as plt
class_acc_array = []
loss_total_array = []
with open('accuracy_epoch.txt','r') as f:
    print('Parsing accuracy file')
    for line in f:
        line_split = line.strip().split(' ')
        (rpn_cls,rpn_regr,detector_cls,detector_regr,class_acc) = line_split
        loss_total = float(rpn_cls)+float(rpn_regr)+float(detector_cls)+float(detector_regr)
        class_acc_array.append(class_acc)
        loss_total_array.append(loss_total)

plt.figure(1)
plt.plot(class_acc_array)
plt.title('accuracy')
plt.xlabel('epoch')
plt.ylabel('precision')
plt.show()


plt.figure(2)
plt.plot(loss_total_array)
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

