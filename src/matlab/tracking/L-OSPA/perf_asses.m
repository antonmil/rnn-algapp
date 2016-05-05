function [dist,loce,carde,card_X,card_Y] = perf_asses(trk,est_trk,OSPA,PLOT)
% est_trk - estimated tracks
% VER 2: : new assignment of estimated to true tracks
%
narginchk(2, 4)

if nargin == 2
    OSPA.p = 1;
    OSPA.c = 25;
    OSPA.l = OSPA.c;
    PLOT = 'Yes';
elseif nargin == 3
    PLOT = 'Yes';
end

[a1,K1,num_trk] = size(trk);
[a2,K2,num_etrk] = size(est_trk);

if (a1~= a2) || (K1 ~= K2) 
    error('wrong input');
end

% find the global best assignment of true tracks to estimated tracks
DELTA = 80;
D = zeros(num_trk,num_etrk);
for i=1:num_trk
    t = trk(:,:,i);
    for j=1:num_etrk
        et = est_trk(:,:,j);
        for k=1:K1
            if not(isnan(t(1,k))) || not(isnan(et(1,k)))
                d = sqrt(sum((t(:,k) - et(:,k)).^2));
                if not(isnan(d))
                    D(i,j) = D(i,j) + min(DELTA,d);
                else
                    D(i,j) = D(i,j) + DELTA;
                end
            end
        end
        D(i,j) = D(i,j)/K1;
    end
end



[Matching,Cost] = Hungarian(D);
Miss_trk = num_etrk;
for i=1:num_trk
    Mach = find(Matching(i,:) == 1);
    if isempty(Mach)
        Miss_trk = Miss_trk +1;
        trk_corr(i)= Miss_trk;
    else
    trk_corr(i) = Mach;
    end
end


% plot 
if strcmp(PLOT,'Yes')
    v = [0.1 0.5 0.9];
    ix = 0;
    for i=1:length(v)
        for j=1:length(v)
            for k=1:length(v);
                ix = ix+1;
                col(ix,:) = [v(i) v(j) v(k)];
            end
        end
    end

    figure(10);
    for i=1:num_trk
        plot(trk(1,:,i),trk(3,:,i),'-','Color',col(2*i,:));
        hold on;
        plot(est_trk(1,:,trk_corr(i)),est_trk(3,:,trk_corr(i)),':','Color',col(2*i,:));
    end
    hold off;
end
% For every k assign labels to estimated tracks and runs OSPA
for k=1:K1
    X = [];
    Xl = [];
    for i=1:num_trk
        if not(isnan(trk(1,k,i)))
            X = [X trk(:,k,i)];
            Xl = [Xl i];
        end
    end
    card_X(k) = size(X,2);
    Y = [];
    Yl = [];
    for i=1:num_etrk
        if not(isnan(est_trk(1,k,i)))
            Y = [Y est_trk(:,k,i)];
            ix =find(trk_corr == i);
            if not(isempty(ix))
                Yl = [Yl ix];
            else
                Yl = [Yl 12345];
            end
        end
    end
    card_Y(k) = size(Y,2);
    %
    [dist(k),loce(k),carde(k)] = trk_ospa_dist(X,Xl,Y,Yl,OSPA);
end

if strcmp(PLOT,'Yes')
    figure(2);
    plot([1:K1],card_X,'r');
    hold on;
    plot([1:K1],card_Y,'b');
    hold off;
    figure(3);
    plot([1:K1],dist,'r','Linewidth',2);
    hold on;
    plot([1:K1],loce,'b--',[1:K1],carde,'k:');
    hold off;
end