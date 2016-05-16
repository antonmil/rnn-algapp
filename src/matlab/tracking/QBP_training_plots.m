%%
clf
hold on


load('D:\Dropbox\research\rnn-algapp\plot_data_0517QBc-1.mat')
% plot(loss_x, train_energies, 'b--')
% plot(loss_x, val_energies, 'b-.')
plot(loss_x, gt_replaced, 'b-')
plot(loss_x, train_mm, 'b.')
plot(loss_x, val_mm, 'bo')

load('D:\Dropbox\research\rnn-algapp\plot_data_0517QBc-2.mat')
% plot(loss_x, train_energies, 'r--')
% plot(loss_x, val_energies, 'r-.')
plot(loss_x, gt_replaced, 'r-')
plot(loss_x, train_mm, 'r.')
plot(loss_x, val_mm, 'ro')
