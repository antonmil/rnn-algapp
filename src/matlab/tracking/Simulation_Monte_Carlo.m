close all
clear all
clc

addpath([pwd,'\L-OSPA'])

K= 20;                                 %number of frames
T= 1;                                   %sampling period [s]
OSPA.p = 2;OSPA.c = 25;OSPA.l = 25;
Num_Exp=100; % number of monte carlo experiments


u_image=30;v_image=30;
Surv_region=[1 u_image; 1 v_image];     %survillance area
X0=[5 1 13 0.2;5 1 15 0;5 1 17 -0.2];   % Inital state

F11=[1 T;0 1];
F(:,:,1)=blkdiag(F11,F11); % The transition matrix for the dynamic model 1
q1=0.0001; % The standard deviation of the process noise for the dynamic model 1
Q11=q1*[T^3/3 T^2/2;T^2/2 T];
Q0(:,:,1)=blkdiag(Q11,Q11); % The process covariance matrix for the dynamic model 1
P0=Q0(:,:,1);

q2=0.02;
Q22=q2*[T^3/3 T^2/2;T^2/2 T];
Q(:,:,1)=blkdiag(Q22,Q22); % The process covariance matrix for the dynamic model 1

% Measurement Matrices
lambda_c=3;% Mean number of clutter per frame

H=[1 0 0 0;0 0 1 0]; % % Measurement matrix
R=[.1 0;0 0.1]; % Measurement covariance matrix
PD=0.7; % Probabilty of detection

JPDA_P=[lambda_c/(u_image*v_image),30];%Beta=2/(u_image*v_image);Gate=30;
S_limit=100;
 
% PD Parameters
PD_Option='Constant'; % PD_Option='State-Dependent';
H_PD=[1 0 0 0;0 0 1 0];

% IMM_Parameters
mui0=1; % The initial values for the IMM probability weights
TPM_Option='Constant'; % TPM_Option='State-Dependent';
TPM=1; % TPM{2,2}=1;TPM{2,1}=2*eye(2,2);TPM{1,2}=1;TPM{1,1}=0.5*eye(2,2);
H_TPM=[0 1 0 0;0 0 0 1];

Tracking_Scheme='JPDA';
TN_H=[1 10:10:100]; % Threshold for considering maximum number of Hypotheses

dist_lospa1_i = zeros(Num_Exp,K);
loce_lospa1_i= zeros(Num_Exp,K);
carde_lospa1_i= zeros(Num_Exp,K);

dist_lospa3_m = zeros(Num_Exp,K,length(TN_H));
loce_lospa3_m= zeros(Num_Exp,K,length(TN_H));
carde_lospa3_m= zeros(Num_Exp,K,length(TN_H));

dist_lospa3_i = zeros(Num_Exp,K);
loce_lospa3_i= zeros(Num_Exp,K);
carde_lospa3_i= zeros(Num_Exp,K);


JPDA_Er=cell(1,length(1:max(TN_H)));
NVHypo=cell(Num_Exp,K);
TNH=cell(Num_Exp,K);
TimeComp=cell(Num_Exp,K);



for NoE=1:Num_Exp
    close all
    clc
    disp(['Experiment = ',num2str(NoE)])
X= gen_state(F,Q0,X0,K);
N_T=max(cellfun(@(x) size(x,2),X));
[Z,iden]= gen_detection2(H,R,X,PD,Surv_region,lambda_c,1);
XYZ=cellfun(@(x) x',Z,'UniformOutput', false);

Obj_num = size(X0,1);
Xdy=cell(1,Obj_num);
Frdy=cell(1,Obj_num);
for k=1:K
    for nn=1:size(X{k},2)
        Xdy{nn}=[Xdy{nn} X{k}(:,nn)];
        if k==1
            Frdy{nn}(1)=1;
        else
          Frdy{nn}(2)=k; 
        end
    end

end


xy_GT = NaN*ones(2,K,Obj_num);
for t=1:K
    for j=1:size(Xdy,2)
        if t>=Frdy{j}(1)&&t<=Frdy{j}(2)
            xy_GT(:,t,j) = [Xdy{j}(1,t-Frdy{j}(1)+1);Xdy{j}(3,t-Frdy{j}(1)+1)];% For L-OSPA
%             plot(xy_GT(1,t,j),xy_GT(2,t,j),'.','Color',colorord(j,:))
%             hold on
        end
    end
    
end

colorord=[1 0 0;0 1 0;0 0 1; 1 1 0;1 0 1;0 1 1];
figure,
for k=1:K
    for nn=1:size(X{k},2)
        plot(X{k}(1,nn),X{k}(3,nn),'s','Color',colorord(nn,:),'MarkerEdgeColor',colorord(nn,:),...
            'MarkerFaceColor',colorord(nn,:),'MarkerSize',4)
        hold on
    end
    axis([Surv_region(1,:) Surv_region(2,:)])
    pause(0.05)
end
daspect([0.7,1,1])
set(gca,'FontName','Times','FontSize',16),
xlabel('x-coordinate'),ylabel('y-coordinate')
set(gca,'YTick',5:10:30),set(gca,'XTick',5:10:30),

% print -dpsc 'GT_State.eps'

figure,

for k=1:K
    ix0=find(iden{k}==0);
    plot(Z{k}(1,ix0),Z{k}(2,ix0),'*k','MarkerSize',6)
    for nn=1:size(X{k},2)
        ix=find(iden{k}==nn);
        if ~isempty(ix)
            plot(Z{k}(1,ix),Z{k}(2,ix),'*','Color',colorord(iden{k}(ix),:),'MarkerEdgeColor',colorord(iden{k}(ix),:),...
            'MarkerFaceColor',colorord(iden{k}(ix),:),'MarkerSize',6)
            hold on
        end
    end

    axis([Surv_region(1,:) Surv_region(2,:)])
    pause(0.05)
end
daspect([0.7,1,1])
set(gca,'FontName','Times','FontSize',16),
xlabel('x-coordinate'),ylabel('y-coordinate')
set(gca,'YTick',5:10:30),set(gca,'XTick',5:10:30),

% print -dpsc 'Detections.eps'
%% 1 Frames, all assignments
% JPDA Parameters
JPDA_multiscale=1; % Time-Frame windows
N_H=inf; % Threshold for considering maximum number of Hypotheses

[XeT,~,~,~,Ff,Term_Con,~]=MULTISCAN_JPDA0(XYZ,F,Q,H,R,X0,P0,Tracking_Scheme,JPDA_P,N_H,...
    JPDA_multiscale,PD,S_limit,mui0,TPM,TPM_Option,H_TPM);

X_size=cellfun(@(x) size(x,2), XeT, 'UniformOutput', false);
Ff=cellfun(@(x,y,z) x(1):x(1)+y-1-z, Ff,X_size,Term_Con, 'ErrorHandler', @errorfun, ...
    'UniformOutput', false);
Ff_size=cellfun(@(x) size(x,2), Ff, 'UniformOutput', false);
XeT=cellfun(@(x,y) x(:,1:y),XeT,Ff_size, 'ErrorHandler', @errorfun, ...
    'UniformOutput', false);

[X_tr_j,Fr_tr_j,~,~]=Trajectory_Generator_Pruner_IMMJPDA(XeT,Ff,K,0);
[~,est_trk_j]=Track_Preprator(X_tr_j,Xdy,Fr_tr_j,Frdy,K);

[dist_lospa1_i(NoE,:),loce_lospa1_i(NoE,:),carde_lospa1_i(NoE,:),~,~] = perf_asses(xy_GT,est_trk_j(1:2:3,:,:),OSPA,'No');
% 
% figure,
%  for k=1:K
%     for n=1:size(XeT,2)
%      idx=find(Ff{1,n}==k);
%      if ~isempty(idx)
%           plot(XeT{1,n}(1,idx),XeT{1,n}(3,idx),'o','Color',colorord(n,:),...
%               'MarkerEdgeColor',colorord(n,:),'MarkerSize',8)
%               hold on
%      end
%     end
% axis([Surv_region(1,:) Surv_region(2,:)])
%     pause(0.05)
%  end
% for nn=1:size(X0,1)
%         Xtrg=zeros(1,K);
%         Ytrg=zeros(1,K);
%     for k=1:K
%         Xtrg(k)=X{k}(1,nn);
%         Ytrg(k)=X{k}(3,nn);
%     end
%     plot(Xtrg,Ytrg,'-','Color',colorord(nn,:),'LineWidth',2)
%     hold on
%     axis([Surv_region(1,:) Surv_region(2,:)])
% end
% daspect([0.7,1,1])
% set(gca,'FontName','Times','FontSize',16),
% xlabel('x-coordinate'),ylabel('y-coordinate')
% set(gca,'YTick',5:10:30),set(gca,'XTick',5:10:30),
% 
% print -dpsc 'JPDA.eps'
% % title(['Hypothesis # = ',num2str(N_H),', Frames # = ',num2str(JPDA_multiscale)])
clear Xtrg Ytrg XeT Ff est_trk_j X_tr_j Fr_tr_j JPDA_multiscale N_H tStart
% 
% %% 3 Frames, m-Best assignment
% JPDA Parameters
JPDA_multiscale=3; % Time-Frame windows
for jm=1:length(TN_H)
[XeT,~,~,~,Ff,Term_Con,~]=MULTISCAN_JPDA0(XYZ,F,Q,H,R,X0,P0,Tracking_Scheme,JPDA_P,TN_H(jm),...
    JPDA_multiscale,PD,S_limit,mui0,TPM,TPM_Option,H_TPM);

X_size=cellfun(@(x) size(x,2), XeT, 'UniformOutput', false);
Ff=cellfun(@(x,y,z) x(1):x(1)+y-1-z, Ff,X_size,Term_Con, 'ErrorHandler', @errorfun, ...
    'UniformOutput', false);
Ff_size=cellfun(@(x) size(x,2), Ff, 'UniformOutput', false);
XeT=cellfun(@(x,y) x(:,1:y),XeT,Ff_size, 'ErrorHandler', @errorfun, ...
    'UniformOutput', false);

[X_tr_j,Fr_tr_j,~,~]=Trajectory_Generator_Pruner_IMMJPDA(XeT,Ff,K,0);
[~,est_trk_j]=Track_Preprator(X_tr_j,Xdy,Fr_tr_j,Frdy,K);

[dist_lospa3_m(NoE,:,jm),loce_lospa3_m(NoE,:,jm),carde_lospa3_m(NoE,:,jm),~,~] = perf_asses(xy_GT,est_trk_j(1:2:3,:,:),OSPA,'No');

clear XeT Ff est_trk_j X_tr_j Fr_tr_j 

end
% 
% figure,
%  for k=1:K
%     for n=1:size(XeT,2)
%      idx=find(Ff{1,n}==k);
%      if ~isempty(idx)
%           plot(XeT{1,n}(1,idx),XeT{1,n}(3,idx),'o','Color',colorord(n,:),...
%               'MarkerEdgeColor',colorord(n,:),'MarkerSize',8)
%               hold on
%      end
%     end
% axis([Surv_region(1,:) Surv_region(2,:)])
%     pause(0.05)
%  end
% for nn=1:size(X0,1)
%         Xtrg=zeros(1,K);
%         Ytrg=zeros(1,K);
%     for k=1:K
%         Xtrg(k)=X{k}(1,nn);
%         Ytrg(k)=X{k}(3,nn);
%     end
%     plot(Xtrg,Ytrg,'-','Color',colorord(nn,:),'LineWidth',2)
%     hold on
%     axis([Surv_region(1,:) Surv_region(2,:)])
% end
% 
% daspect([0.7,1,1])
% set(gca,'FontName','Times','FontSize',16),
% xlabel('x-coordinate'),ylabel('y-coordinate')
% set(gca,'YTick',5:10:30),set(gca,'XTick',5:10:30),
% 
% print -dpsc '3-JPDA-1.eps'
% % title(['Hypothesis # = ',num2str(N_H),', Frames # = ',num2str(JPDA_multiscale)])
clear Xtrg Ytrg XeT Ff est_trk_j X_tr_j Fr_tr_j JPDA_multiscale N_H tStart
% 
% %% 3 Frames, 50-Best assignments
% % JPDA Parameters
% JPDA_multiscale=3; % Time-Frame windows
% N_H=50; % Threshold for considering maximum number of Hypotheses
% 
% tStart = tic; 
% [XeT,~,~,~,Ff,Term_Con,~]=MULTISCAN_JPDA0(XYZ,F,Q,H,R,X0,P0,Tracking_Scheme,JPDA_P,N_H,...
%     JPDA_multiscale,PD,S_limit,mui0,TPM,TPM_Option,H_TPM);
% tElapsed3_50(NoE) = toc(tStart);
% 
% X_size=cellfun(@(x) size(x,2), XeT, 'UniformOutput', false);
% Ff=cellfun(@(x,y,z) x(1):x(1)+y-1-z, Ff,X_size,Term_Con, 'ErrorHandler', @errorfun, ...
%     'UniformOutput', false);
% Ff_size=cellfun(@(x) size(x,2), Ff, 'UniformOutput', false);
% XeT=cellfun(@(x,y) x(:,1:y),XeT,Ff_size, 'ErrorHandler', @errorfun, ...
%     'UniformOutput', false);
% 
% [X_tr_j,Fr_tr_j,~,~]=Trajectory_Generator_Pruner_IMMJPDA(XeT,Ff,K,0);
% [~,est_trk_j]=Track_Preprator(X_tr_j,Xdy,Fr_tr_j,Frdy,K);
% 
% [dist_lospa3_50(NoE,:),loce_lospa3_50(NoE,:),carde_lospa3_50(NoE,:),~,~] = perf_asses(xy_GT,est_trk_j(1:2:3,:,:),OSPA,'No');
% 
% figure,
%  for k=1:K
%     for n=1:size(XeT,2)
%      idx=find(Ff{1,n}==k);
%      if ~isempty(idx)
%           plot(XeT{1,n}(1,idx),XeT{1,n}(3,idx),'o','Color',colorord(n,:),...
%               'MarkerEdgeColor',colorord(n,:),'MarkerSize',8)
%               hold on
%      end
%     end
% axis([Surv_region(1,:) Surv_region(2,:)])
%     pause(0.05)
%  end
% for nn=1:size(X0,1)
%         Xtrg=zeros(1,K);
%         Ytrg=zeros(1,K);
%     for k=1:K
%         Xtrg(k)=X{k}(1,nn);
%         Ytrg(k)=X{k}(3,nn);
%     end
%     plot(Xtrg,Ytrg,'-','Color',colorord(nn,:),'LineWidth',2)
%     hold on
%     axis([Surv_region(1,:) Surv_region(2,:)])
% end
% daspect([0.7,1,1])
% set(gca,'FontName','Times','FontSize',16),
% xlabel('x-coordinate'),ylabel('y-coordinate')
% set(gca,'YTick',5:10:30),set(gca,'XTick',5:10:30),
% 
% print -dpsc '3-JPDA-50.eps'
% %title(['Hypothesis # = ',num2str(N_H),', Frames # = ',num2str(JPDA_multiscale)])
% clear Xtrg Ytrg XeT Ff est_trk_j X_tr_j Fr_tr_j JPDA_multiscale N_H tStart
% 
% %% 3 Frames, all assignments
% % JPDA Parameters
% JPDA_multiscale=3; % Time-Frame windows
% N_H=inf; % Threshold for considering maximum number of Hypotheses
% 
% tStart = tic; 
% [XeT,~,~,~,Ff,Term_Con,~]=MULTISCAN_JPDA0(XYZ,F,Q,H,R,X0,P0,Tracking_Scheme,JPDA_P,N_H,...
%     JPDA_multiscale,PD,S_limit,mui0,TPM,TPM_Option,H_TPM);
% tElapsed3_i(NoE) = toc(tStart);
% 
% X_size=cellfun(@(x) size(x,2), XeT, 'UniformOutput', false);
% Ff=cellfun(@(x,y,z) x(1):x(1)+y-1-z, Ff,X_size,Term_Con, 'ErrorHandler', @errorfun, ...
%     'UniformOutput', false);
% Ff_size=cellfun(@(x) size(x,2), Ff, 'UniformOutput', false);
% XeT=cellfun(@(x,y) x(:,1:y),XeT,Ff_size, 'ErrorHandler', @errorfun, ...
%     'UniformOutput', false);
% 
% [X_tr_j,Fr_tr_j,~,~]=Trajectory_Generator_Pruner_IMMJPDA(XeT,Ff,K,0);
% [~,est_trk_j]=Track_Preprator(X_tr_j,Xdy,Fr_tr_j,Frdy,K);
% 
% [dist_lospa3_i(NoE,:),loce_lospa3_i(NoE,:),carde_lospa3_i(NoE,:),~,~] = perf_asses(xy_GT,est_trk_j(1:2:3,:,:),OSPA,'No');
% 
% figure,
%  for k=1:K
%     for n=1:size(XeT,2)
%      idx=find(Ff{1,n}==k);
%      if ~isempty(idx)
%           plot(XeT{1,n}(1,idx),XeT{1,n}(3,idx),'o','Color',colorord(n,:),...
%               'MarkerEdgeColor',colorord(n,:),'MarkerSize',8)
%               hold on
%      end
%     end
% axis([Surv_region(1,:) Surv_region(2,:)])
%     pause(0.05)
%  end
% for nn=1:size(X0,1)
%         Xtrg=zeros(1,K);
%         Ytrg=zeros(1,K);
%     for k=1:K
%         Xtrg(k)=X{k}(1,nn);
%         Ytrg(k)=X{k}(3,nn);
%     end
%     plot(Xtrg,Ytrg,'-','Color',colorord(nn,:),'LineWidth',2)
%     hold on
%     axis([Surv_region(1,:) Surv_region(2,:)])
% end
% daspect([0.7,1,1])
% set(gca,'FontName','Times','FontSize',16),
% xlabel('x-coordinate'),ylabel('y-coordinate')
% set(gca,'YTick',5:10:30),set(gca,'XTick',5:10:30),
% 
% print -dpsc '3-JPDA.eps'
% % title(['Hypothesis # = ',num2str(N_H),', Frames # = ',num2str(JPDA_multiscale)])
% clear Xtrg Ytrg XeT Ff est_trk_j X_tr_j Fr_tr_j JPDA_multiscale N_H tStart

%% 3 Frames, probabilty comparison mbest Vs all assignments
% JPDA Parameters
JPDA_multiscale=3; % Time-Frame windows

[XeT,~,~,~,Ff,Term_Con,~,diff,NVHypo(NoE,:),TNH(NoE,:),TimeComp(NoE,:)]=MULTISCAN_JPDA_Comparison(XYZ,F,Q,H,R,X0,P0,Tracking_Scheme,JPDA_P,TN_H,...
    JPDA_multiscale,PD,S_limit,mui0,TPM,TPM_Option,H_TPM);

X_size=cellfun(@(x) size(x,2), XeT, 'UniformOutput', false);
Ff=cellfun(@(x,y,z) x(1):x(1)+y-1-z, Ff,X_size,Term_Con, 'ErrorHandler', @errorfun, ...
    'UniformOutput', false);
Ff_size=cellfun(@(x) size(x,2), Ff, 'UniformOutput', false);
XeT=cellfun(@(x,y) x(:,1:y),XeT,Ff_size, 'ErrorHandler', @errorfun, ...
    'UniformOutput', false);

[X_tr_j,Fr_tr_j,~,~]=Trajectory_Generator_Pruner_IMMJPDA(XeT,Ff,K,0);
[~,est_trk_j]=Track_Preprator(X_tr_j,Xdy,Fr_tr_j,Frdy,K);

[dist_lospa3_i(NoE,:),loce_lospa3_i(NoE,:),carde_lospa3_i(NoE,:),~,~] = perf_asses(xy_GT,est_trk_j(1:2:3,:,:),OSPA,'No');


diff(cellfun(@isempty,diff))=[];
for kj=1:max(TN_H)
err_at_NH=cellfun(@(x) x{kj},diff,'UniformOutput', false);
err_at_NH(cellfun(@isempty,err_at_NH))=[];
JPDA_Er{1,kj}=[JPDA_Er{1,kj};cell2mat(err_at_NH)];
end
clear  XeT Ff est_trk_j X_tr_j Fr_tr_j JPDA_multiscale N_H diff err_at_NH
save('synthetic_results_new','JPDA_Er','NVHypo','TNH','TimeComp',...
    'dist_lospa1_i','loce_lospa1_i','carde_lospa1_i',...
    'dist_lospa3_m','loce_lospa3_m','carde_lospa3_m',...
    'dist_lospa3_i','loce_lospa3_i','carde_lospa3_i')

end



Mmx=0;noe=0;indxx=0;
for NoE=1:Num_Exp
for i=2:20
    [mmx,indx]=max((cellfun(@(x) x(end),TimeComp{NoE,i}(1,:))));
    if mmx>=Mmx
        Mmx=mmx
        noe=NoE
        indxx=indx
        II=i
        mbestsol=cellfun(@(x) x(end),TimeComp{NoE,i}(2,indx))
    end
end
end

ttme=[];tjpda=[];jnh=0;
for NoE=1:Num_Exp
for i=2:20
    for jh=1:size(TimeComp{NoE,i}(2,:),2)
        if size(TimeComp{NoE,i}{2,jh},2)==max(TN_H)
        ttme=[ttme;TimeComp{NoE,i}{2,jh}];
        end
        if size(TimeComp{NoE,i}{1,jh},2)>10000
           tjpda=[tjpda;TimeComp{NoE,i}{1,jh}(end)];
           jnh=[jnh;size(TimeComp{NoE,i}{1,jh},2)];
        end 
    end
end
end

Meann=zeros(1,max(TN_H));
Minn=zeros(1,max(TN_H));
Maxx=zeros(1,max(TN_H));
Stdd=zeros(1,max(TN_H));
mxS=max(cellfun(@(x) size(x,1),JPDA_Er));
BxPlot=NaN*ones(mxS,size(JPDA_Er,2));
for kj=1:max(TN_H)

Meann(kj)=mean(JPDA_Er{1,kj}(:,1));
Minn(kj)=min(JPDA_Er{1,kj}(:,1));
Maxx(kj)=max(JPDA_Er{1,kj}(:,1));
Stdd(kj)=std(JPDA_Er{1,kj}(:,1));
BxPlot(1:size(JPDA_Er{1,kj},1),kj)=JPDA_Er{1,kj}(:,1);
end
close all

fig1000=errorbar(1:max(TN_H),Meann,Stdd);
axis([0 100 0 0.42])
set(gca,'YTick',0:0.05:1.2),
set(gca,'FontName','Times','FontSize',20),
figc = get(fig1000, 'Children');
set(figc(1),'color','k','LineWidth',2)
set(figc(2),'color',[0.5 0.5 1],'LineStyle','-','LineWidth',2)

hold on
[hAx,hLine1,hLine2] =plotyy(1:max(TN_H),Meann,1:max(TN_H),mean(ttme,1));
%xlabel('Number of {\it m}-best solutions ')
axis([0 100 0 0.42])
set(hAx(1),'YTick',0:0.05:1.2),
set(hAx(2),'YTick',0:0.5:2),
set(hLine1,'LineWidth',2.5,'color','k')
set(hLine2,'LineWidth',2.5)
set(hAx(1), 'box', 'off')


ylabel(hAx(1),'Data assoociation error') % left y-axis
set(hAx(1),'FontName','Times','FontSize',20),
set(hAx(2),'FontName','Times','FontSize',20),
ylabel(hAx(2),'Processing time (sec.)') % right y-axis

%hold on
%plot(1:max(TN_H),0.01*ones(1,length(1:max(TN_H))),'--','LineWidth',2.5,'color',[0.3 0.3 0.3])

legend('std error','mean error','time','Location','NorthWest')
legend('boxoff')

print -dpsc 'err_box.eps'





close all
figure, 
plot(TN_H, permute(mean(mean(loce_lospa3_m,2),1),[3 1 2]),'-ob','LineWidth',2.5)
hold on
plot(TN_H,(mean2(loce_lospa3_i))*ones(1,11),'--sk','LineWidth',2.5,'MarkerSize',10)
set(gca,'FontName','Times','FontSize',20),
set(gca,'YTick',0.5:0.1:1.4),
axis([0 100 0.5 1.4])
legend('3F-JPDA_{\it m}','3F-JPDA','Location','NorthWest')
%legend('boxoff')
set(gca,'box','off')
ylabel('Location error')
xlabel('Number of {\it m}-best solutions ')
grid on
print -dpsc 'Tracking_accuracy_err.eps'

% disp(['T-OSPA_1_1= ',num2str(mean2(dist_lospa1_1))])
% disp(['T-OSPA_1_10= ',num2str(mean2(dist_lospa1_10))])
disp(['T-OSPA_1_i= ',num2str(mean2(dist_lospa1_i))])
% disp(['T-OSPA_3_1= ',num2str(mean2(dist_lospa3_1))])
% disp(['T-OSPA_3_50= ',num2str(mean2(dist_lospa3_50))])
disp(['T-OSPA_3_i= ',num2str(mean2(dist_lospa3_i))])









