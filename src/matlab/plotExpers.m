function plotExpers(name)

tmpDir = '/home/amilan/Dropbox/research/rnn-algapp/tmp';
if nargin<1, name = '0229A'; end
mask = [name,'*.png'];
rows = 5; cols = 3;
pad = 10;


graphs = dir(fullfile(tmpDir,mask));
oneIm = double(imread(fullfile(tmpDir, graphs(1).name)))/255;
[imX,imY,imC]=size(oneIm);

canvas = ones((imX+pad)*rows, (imY+pad)*cols, 3);

cc=0;
for x=1:rows
    xx=(x-1)*(imX+pad)+1;
    for y=1:cols
        cc=cc+1;
        if cc>length(graphs), break; end
        try oneIm = double(imread(fullfile(tmpDir, graphs(cc).name)))/255;
        catch err, oneIm = ones(imX,imY,imC); end
        yy=(y-1)*(imY+pad)+1;
        imDimX = xx:xx+imX-1;
        imDimY = yy:yy+imY-1;
        canvas(imDimX,imDimY,:) = oneIm;
    end
end

if ~isdir('parsearch'), mkdir('parsearch'); end
filename=['parsearch',filesep,'hps-',name,'.png'];
imwrite(canvas,filename);

% if isdir('/home/amilan/'), imshow(canvas); end
copyfile(filename,'~/Dropbox/research/rnn-algapp/tmp/')

