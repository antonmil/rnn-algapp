%%
% 2. Get a handle to the current figure:
close all
addpath('../external/export_fig/');
addpath('..');

open('ObjVsLOss_TSP.fig')

lineObjs = findobj(gca,'Type','line');

xdata = get(lineObjs, 'XData');  %data from low-level grahics objects
ydata = get(lineObjs, 'YData');


% apend eopch 7
% xdata{1}=[xdata{1}, 7];
% xdata{2}=[xdata{2}, 7];
% ydata{1}=[ydata{1}, 3.47];
% ydata{2}=[ydata{2}, 3.51];


clf
%%
open('Prediction_better_thn_trainset.fig');

lineObjs = findobj(gca,'Type','line');

xdata2 = get(lineObjs, 'XData');  %data from low-level grahics objects
ydata2 = get(lineObjs, 'YData');

% cut to 6 epochs
xdata2=xdata2(1:6);
ydata2=ydata2(1:6);

close 

figure
%%
clf;
hold on

ms = 50;fs = 55;lw=4;

xlim([0 7]);

yyaxis left
plot(xdata{2}, ydata{2}, 'b--','MarkerSize',ms,'linewidth',lw); % loss-based
plot(xdata{1}, ydata{1}, 'b.-','MarkerSize',ms,'linewidth',lw); % obj-based
% plot(xdata{1}, 3.0882*ones(1,length(xdata{1})));

ylabel('Tour length');
ax = gca;
ax.YColor = 'b';

yyaxis right
plot(xdata2, ydata2*100, 'r.-','MarkerSize',ms,'linewidth',lw)
ylabel('% Improved train set');

set(gca, 'FontSize',fs);


box on
grid
xlabel('Epoch')
title('TSP Training');

legend('Loss-based [21]','Loss+Objective-based (ours)','Improved training examples','Location','NorthWest');


export_fig(fullfile(getPaperDir,'figures','TSP-training.pdf'),'-a1','-native','-transparent')