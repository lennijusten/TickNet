import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Author: Lennart Justen
# Data: 11/21/2020

# Calculate confidence thresholds based on Villon et al. 2020

# Villon, S., Mouillot, D., Chaumont, M. et al. A new method to control error rates in automated species identification
# with deep learning algorithms. Sci Rep 10, 10972 (2020). https://doi.org/10.1038/s41598-020-67573-7

df = pd.read_csv('/Users/lenni/Documents/PycharmProjects/MCEVBD Fellowship/Model_Final/Val_final_results.csv')

tau = np.linspace(30, 100, num=5000)

aa = len(df.loc[df['class_num']==0])
dv = len(df.loc[df['class_num']==1])
ix = len(df.loc[df['class_num']==2])

cc_a = []
mc_a = []
uc_a = []
cc_d = []
mc_d = []
uc_d = []
cc_i = []
mc_i = []
uc_i = []
for t in tau:
    cc_a.append(len(df.loc[(df['class_num']==0) & (df['prob']>t) & (df['pred_num']==0)])/aa)
    mc_a.append(len(df.loc[(df['class_num']==0) & (df['prob']>t) & (df['pred_num']!=0)])/aa)
    uc_a.append(len(df.loc[(df['class_num']==0) & (df['prob']<t)])/aa)

    cc_d.append(len(df.loc[(df['class_num']==1) & (df['prob']>t) & (df['pred_num']==1)])/dv)
    mc_d.append(len(df.loc[(df['class_num']==1) & (df['prob']>t) & (df['pred_num']!=1)])/dv)
    uc_d.append(len(df.loc[(df['class_num']==1) & (df['prob']<t)])/dv)

    cc_i.append(len(df.loc[(df['class_num']==2) & (df['prob']>t) & (df['pred_num']==2)])/ix)
    mc_i.append(len(df.loc[(df['class_num']==2) & (df['prob']>t) & (df['pred_num']!=2)])/ix)
    uc_i.append(len(df.loc[(df['class_num']==2) & (df['prob']<t)])/ix)

total_aa = cc_a+mc_a+uc_a
total_dv = cc_d+mc_d+uc_d
total_ix = cc_i+mc_i+uc_i

# plt.plot(tau, cc_d, label='CC')
# plt.plot(tau, mc_d, label='MC')
# plt.plot(tau, uc_d, label='UC')
# plt.plot(tau, np.array(cc_d)+np.array(mc_d), label='CC+MC')
# plt.plot(tau, np.array(cc_d)+np.array(mc_d)+np.array(uc_d), '-', label = 'CC+MC+UC')
# plt.xlabel('tau')
# plt.legend()
# plt.show()


def goal_one(tau, cc, mc, uc, n, label):
    maxval = max(cc)
    indices = [index for index, val in enumerate(cc) if val == maxval]
    mc_indices = [mc[i] for i in indices]

    key = mc_indices.index(min(mc_indices))
    thresh = tau[key]

    print('-----------------------------------------------------')
    # print('Goal 1: Keep best CC while reducing MC rate')
    print('LABEL: {}'.format(label))
    print('Initial accuracy={}'.format(cc[0]))
    print('Final accuracy (excluding unsure)={}'.format(((cc[key])*n/(n-(uc[key]*n)))))
    print('Accuracy increase={}'.format(((cc[key]) * n / (n - (uc[key] * n)))-cc[0]))
    print('Threshold={}'.format(thresh))
    print('CC w/o threshold={}'.format(cc[0]))
    print('MC w/o threshold={}'.format(mc[0]))
    print('--------------')
    print('CC w/ threshold={}'.format(cc[key]))
    print('MC w/ threshold={}'.format(mc[key]))
    print('UC w/ threshold={}'.format(uc[key]))
    print("Number of incorrect images moved to 'unsure': {}/{}".format(uc[key]*n, n))



# goal_one(tau, cc_a, mc_a, uc_a, aa, 'Amblyomma')
# goal_one(tau, cc_d, mc_d, uc_d, dv, 'Dermacentor')
# goal_one(tau, cc_i, mc_i, uc_i, ix, 'Ixodes')


def goal_two(tau, cc, mc, uc, n, label, val):
    indices = np.where(np.array(mc[0:len(mc)-1]) < val)
    indices = indices[0]

    if len(indices)==1:
        min_mc = min(mc[0:len(mc)-1])
        indices = [index for index, val in enumerate(mc[0:len(mc)-1]) if val == min_mc]

    cc_indices = [cc[i] for i in indices]
    key1 = cc_indices.index(max(cc_indices))
    key2 = indices[key1]
    thresh = tau[key2]

    print('-----------------------------------------------------')
    print('Goal 2: Constrain the misclassification error rate to an upper bound while maximizing the correct classification rate')
    print('LABEL: {}'.format(label))
    print('Initial accuracy={}'.format(cc[0]))
    print('Final accuracy (excluding unsure)={}'.format(((cc[key2]) * n / (n - (uc[key2] * n)))))
    print('Accuracy increase={}'.format(((cc[key2]) * n / (n - (uc[key2] * n)))-cc[0]))
    print('Threshold={}'.format(thresh))
    print('CC w/o threshold={}'.format(cc[0]))
    print('MC w/o threshold={}'.format(mc[0]))
    print('--------------')
    print('CC w/ threshold={}'.format(cc[key2]))
    print('MC w/ threshold={}'.format(mc[key2]))
    print('UC w/ threshold={}'.format(uc[key2]))
    print("Number of incorrect images moved to 'unsure': {}/{}".format(uc[key2] * n, n))


def goal_three(): # Not yet implemented
    print('Goal 3: keep the lowest misclassification rate while raising the correct classification error rate (implying a lower coverage)')
    pass


goal_two(tau, cc_a, mc_a, uc_a, aa, 'Amblyomma', val=0.02)
goal_two(tau, cc_d, mc_d, uc_d, dv, 'Dermacentor', val=0.02)
goal_two(tau, cc_i, mc_i, uc_i, ix, 'Ixodes', val=0.02)




