function genFig(mInd, smpl, allFs, allGphsSubSel, randSmpl, asgT, allRes)
%%
clf

Fs = allFs{randSmpl};
gphs = allGphsSubSel{randSmpl};

[im1H, im1W, ~] = size(Fs{1});
[im2H, im2W, ~] = size(Fs{2});
brdWidth = 10; % border width in px

% downsize bigger one
[mv,mi]= max([im1H,im2H]);
scale = min(im1H, im2H)/mv;
Fs{mi} = imresize(Fs{mi}, scale);
% [mv,mi]= max([im1H,im2H]);
[im1H, im1W, ~] = size(Fs{1});
[im2H, im2W, ~] = size(Fs{2});
gphs{mi}.Pt = gphs{2}.Pt * scale;


vshift1 = 0; vshift2 = 0; % vertical shift
if mi==1, vshift2 = abs(im1H-im2H)/2; else vshift1 = abs(im1H-im2H)/2; end
hshift2 = im1W+brdWidth;
canvH = im1H; canvW = im1W+im2W+brdWidth;
canvas = ones(canvH, canvW, 3, 'uint8')*255;
canvas(1+vshift1:im1H+vshift1, 1:im1W, :) = Fs{1};
canvas(:, im1W+1:im1W+brdWidth-1, :) = 255;
canvas(1+vshift2:im2H+vshift2, hshift2:im1W+brdWidth+im2W-1, :) = Fs{2};

% add footer
footer = 60;
canvas(end+1:end+footer,:,:) = 255;


% bgim = allFs{randSmpl}{1};
% bgim = imrotate(bgim,90);
% bgim = flipud(bgim);
imshow(canvas)
hold on

if mi==1, gphs{2}.Pt(1,:) = gphs{2}.Pt(1,:) + vshift2;
else gphs{1}.Pt(1,:) = gphs{1}.Pt(1,:) + vshift1;
end
gphs{2}.Pt(2,:) = gphs{2}.Pt(2,:) + hshift2;



plot(gphs{1}.Pt(2,:),gphs{1}.Pt(1,:),'r.','MarkerSize',30)
plot(gphs{2}.Pt(2,:),gphs{2}.Pt(1,:),'r.','MarkerSize',30)

resMat = allRes{mInd}.resMat(:,:,end);
for n=1:size(resMat,1)
    mtchd = find(resMat(n,:));
    for m=mtchd
        col = 'y';
        if resMat(n,m) ~= asgT.X(n,m), col='b'; end
        lineX = [gphs{1}.Pt(2,n), gphs{2}.Pt(2,m)];
        lineY = [gphs{1}.Pt(1,n), gphs{2}.Pt(1,m)];        
        line(lineX, lineY, 'color', col,'linewidth',3)
    end
%     text(gphs{1}.Pt(2,n),gphs{1}.Pt(1,n),sprintf('%d',n),'color','w');
%     text(gphs{2}.Pt(2,n),gphs{2}.Pt(1,n),sprintf('%d',n),'color','w');    
end
% accuracy and objective
% text(10,20,sprintf('%s', allRes{mInd}.name),'color','w','FontSize',25,'FontWeight','bold');
% text(10,im1H+20,sprintf('Acc: %.2f', allRes{mInd}.acc(end)),'color','k','FontSize',20,'FontWeight','bold');
% text(10,im1H-50,sprintf('Obj: %.2f', allRes{mInd}.obj(end)),'color','k','FontSize',20,'FontWeight','bold');

text(10,im1H+20,sprintf('%s:   Accuracy: %.2f   Objective: %.2f', allRes{mInd}.name,allRes{mInd}.acc(end), allRes{mInd}.obj(end)), ...
    'color','k','FontSize',22,'FontWeight','normal');

%%
method = allRes{mInd}.name;
fname = sprintf('%s/figures/matching-%s-%d.jpg',getPaperDir,method, smpl); 
export_fig(fname,'-a1','-native','-transparent')