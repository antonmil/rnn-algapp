%%
clf
hold on

addpath(genpath('external'));

load('../../tmp/plot_data_0509QAc-1.mat')
plot(loss_x, train_energies, 'b--')
% plot(loss_x, val_energies, 'b-.')
% plot(loss_x, gt_replaced, 'b-')
% plot(loss_x, train_mm, 'b.')
% plot(loss_x, val_mm, 'bo')

load('../../tmp/plot_data_0509QAc-2.mat')
plot(loss_x, train_energies, 'r--')
% plot(loss_x, val_energies, 'r-.')
% plot(loss_x, gt_replaced, 'r-')
% plot(loss_x, train_mm, 'r.')
% plot(loss_x, val_mm, 'ro')

box on
xlabel('# iteration');
ylabel('% GT replaced');
legend('loss-based','objective-based');

% export_fig(sprintf('gt-replaced.pdf',NoE),'-a1','-native','-transparent');