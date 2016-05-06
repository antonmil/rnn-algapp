close all
clear all
clc

addpath(fullfile([pwd,filesep,'L-OSPA']))

rng(3);
K= 20;                                 %number of frames
T= 1;                                   %sampling period [s]
OSPA.p = 2;OSPA.c = 25;OSPA.l = 25;
Num_Exp=100; % number of monte carlo experiments


u_image=30;v_image=30;
Surv_region=[1 u_image; 1 v_image];     %survillance area
X0=[5 1 11 0.4;5 1 13 0.2;5 1 15 0;5 1 17 -0.2;5 1 19 -0.4];   % Inital state

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
lambda_c=0;% Mean number of clutter per frame

H=[1 0 0 0;0 0 1 0]; % % Measurement matrix
R=[.1 0;0 0.1]; % Measurement covariance matrix
PD=1; % Probabilty of detection

JPDA_P=[lambda_c/(u_image*v_image),inf];%Beta=2/(u_image*v_image);Gate=30;
S_limit=inf;
 
% PD Parameters
PD_Option='Constant'; % PD_Option='State-Dependent';
H_PD=[1 0 0 0;0 0 1 0];

% IMM_Parameters
mui0=1; % The initial values for the IMM probability weights
TPM_Option='Constant'; % TPM_Option='State-Dependent';
TPM=1; % TPM{2,2}=1;TPM{2,1}=2*eye(2,2);TPM{1,2}=1;TPM{1,1}=0.5*eye(2,2);
H_TPM=[0 1 0 0;0 0 0 1];

TN_H=[1 10:10:100]; % Threshold for considering maximum number of Hypotheses

dist_lospa1_jpda = zeros(Num_Exp,K);
loce_lospa1_jpda= zeros(Num_Exp,K);
carde_lospa1_jpda= zeros(Num_Exp,K);

dist_lospa1_jpda_ha = zeros(Num_Exp,K);
loce_lospa1_jpda_ha= zeros(Num_Exp,K);
carde_lospa1_jpda_ha= zeros(Num_Exp,K);

dist_lospa1_ha = zeros(Num_Exp,K);
loce_lospa1_ha= zeros(Num_Exp,K);
carde_lospa1_ha= zeros(Num_Exp,K);

dist_lospa1_LSTM = zeros(Num_Exp,K);
loce_lospa1_LSTM= zeros(Num_Exp,K);
carde_lospa1_LSTM= zeros(Num_Exp,K);

dist_lospa1_LSTM_ha = zeros(Num_Exp,K);
loce_lospa1_LSTM_ha= zeros(Num_Exp,K);
carde_lospa1_LSTM_ha= zeros(Num_Exp,K);



JPDA_Er=cell(1,length(1:max(TN_H)));
NVHypo=cell(Num_Exp,K);
TNH=cell(Num_Exp,K);
TimeComp=cell(Num_Exp,K);

JPDA_Er_3F=cell(1,length(1:max(TN_H)));
NVHypo_3F=cell(Num_Exp,K);
TNH_3F=cell(Num_Exp,K);
TimeComp_3F=cell(Num_Exp,K);



for NoE=1:Num_Exp
    close all
    clc
    disp(['Experiment = ',num2str(NoE)])
X= gen_state(F,Q0,X0,K);
N_T=max(cellfun(@(x) size(x,2),X));
[Z,iden]= gen_detection2(H,R,X,PD,Surv_region,lambda_c,1);
XYZ=cellfun(@(x) x',Z,'UniformOutput', false);

save([pwd,filesep,'Data',filesep,'Experiment_',num2str(NoE)],'X','Z','iden','XYZ')

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
%% 1 Frame, JPDA
% JPDA Parameters
JPDA_multiscale=1; % Time-Frame windows
Tracking_Scheme='JPDA';
[XeT,~,~,~,Ff,Term_Con,~]=MULTISCAN_JPDA0(XYZ,F,Q,H,R,X0,P0,Tracking_Scheme,JPDA_P,TN_H(end),...
    JPDA_multiscale,PD,S_limit,mui0,TPM,TPM_Option,H_TPM);

X_size=cellfun(@(x) size(x,2), XeT, 'UniformOutput', false);
Ff=cellfun(@(x,y,z) x(1):x(1)+y-1-z, Ff,X_size,Term_Con, 'ErrorHandler', @errorfun, ...
    'UniformOutput', false);
Ff_size=cellfun(@(x) size(x,2), Ff, 'UniformOutput', false);
XeT=cellfun(@(x,y) x(:,1:y),XeT,Ff_size, 'ErrorHandler', @errorfun, ...
    'UniformOutput', false);

[X_tr_j,Fr_tr_j,~,~]=Trajectory_Generator_Pruner_IMMJPDA(XeT,Ff,K,0);
[~,est_trk_j]=Track_Preprator(X_tr_j,Xdy,Fr_tr_j,Frdy,K);

figure,
    for nn=1:size(X_tr_j,2)
        plot(X_tr_j{nn}(1,:),X_tr_j{nn}(3,:),'s','Color',colorord(nn,:),'MarkerEdgeColor',colorord(nn,:),...
            'MarkerFaceColor',colorord(nn,:),'MarkerSize',4)
        hold on
    end
    axis([Surv_region(1,:) Surv_region(2,:)])


[dist_lospa1_jpda(NoE,:),loce_lospa1_jpda(NoE,:),carde_lospa1_jpda(NoE,:),~,~] = perf_asses(xy_GT,est_trk_j(1:2:3,:,:),OSPA,'No');
clear Xtrg Ytrg XeT Ff est_trk_j X_tr_j Fr_tr_j JPDA_multiscale N_H tStart
%% 1 Frame, JPDA_HA
% JPDA Parameters
JPDA_multiscale=1; % Time-Frame windows
Tracking_Scheme='JPDA_HA';
[XeT,~,~,~,Ff,Term_Con,~]=MULTISCAN_JPDA0(XYZ,F,Q,H,R,X0,P0,Tracking_Scheme,JPDA_P,TN_H(end),...
    JPDA_multiscale,PD,S_limit,mui0,TPM,TPM_Option,H_TPM);

X_size=cellfun(@(x) size(x,2), XeT, 'UniformOutput', false);
Ff=cellfun(@(x,y,z) x(1):x(1)+y-1-z, Ff,X_size,Term_Con, 'ErrorHandler', @errorfun, ...
    'UniformOutput', false);
Ff_size=cellfun(@(x) size(x,2), Ff, 'UniformOutput', false);
XeT=cellfun(@(x,y) x(:,1:y),XeT,Ff_size, 'ErrorHandler', @errorfun, ...
    'UniformOutput', false);

[X_tr_j,Fr_tr_j,~,~]=Trajectory_Generator_Pruner_IMMJPDA(XeT,Ff,K,0);
[~,est_trk_j]=Track_Preprator(X_tr_j,Xdy,Fr_tr_j,Frdy,K);

figure,
    for nn=1:size(X_tr_j,2)
        plot(X_tr_j{nn}(1,:),X_tr_j{nn}(3,:),'s','Color',colorord(nn,:),'MarkerEdgeColor',colorord(nn,:),...
            'MarkerFaceColor',colorord(nn,:),'MarkerSize',4)
        hold on
    end
    axis([Surv_region(1,:) Surv_region(2,:)])


[dist_lospa1_jpda_ha(NoE,:),loce_lospa1_jpda_ha(NoE,:),carde_lospa1_jpda_ha(NoE,:),~,~] = perf_asses(xy_GT,est_trk_j(1:2:3,:,:),OSPA,'No');
clear Xtrg Ytrg XeT Ff est_trk_j X_tr_j Fr_tr_j JPDA_multiscale N_H tStart
%% 1 Frame, HA
% JPDA Parameters
JPDA_multiscale=1; % Time-Frame windows
Tracking_Scheme='HA';
[XeT,~,~,~,Ff,Term_Con,~]=MULTISCAN_JPDA0(XYZ,F,Q,H,R,X0,P0,Tracking_Scheme,JPDA_P,TN_H(end),...
    JPDA_multiscale,PD,S_limit,mui0,TPM,TPM_Option,H_TPM);

X_size=cellfun(@(x) size(x,2), XeT, 'UniformOutput', false);
Ff=cellfun(@(x,y,z) x(1):x(1)+y-1-z, Ff,X_size,Term_Con, 'ErrorHandler', @errorfun, ...
    'UniformOutput', false);
Ff_size=cellfun(@(x) size(x,2), Ff, 'UniformOutput', false);
XeT=cellfun(@(x,y) x(:,1:y),XeT,Ff_size, 'ErrorHandler', @errorfun, ...
    'UniformOutput', false);

[X_tr_j,Fr_tr_j,~,~]=Trajectory_Generator_Pruner_IMMJPDA(XeT,Ff,K,0);
[~,est_trk_j]=Track_Preprator(X_tr_j,Xdy,Fr_tr_j,Frdy,K);

figure,
    for nn=1:size(X_tr_j,2)
        plot(X_tr_j{nn}(1,:),X_tr_j{nn}(3,:),'s','Color',colorord(nn,:),'MarkerEdgeColor',colorord(nn,:),...
            'MarkerFaceColor',colorord(nn,:),'MarkerSize',4)
        hold on
    end
    axis([Surv_region(1,:) Surv_region(2,:)])


[dist_lospa1_ha(NoE,:),loce_lospa1_ha(NoE,:),carde_lospa1_ha(NoE,:),~,~] = perf_asses(xy_GT,est_trk_j(1:2:3,:,:),OSPA,'No');
clear Xtrg Ytrg XeT Ff est_trk_j X_tr_j Fr_tr_j JPDA_multiscale N_H tStart
%% 1 Frame, LSTM
% JPDA Parameters
JPDA_multiscale=1; % Time-Frame windows
Tracking_Scheme='LSTM';
[XeT,~,~,~,Ff,Term_Con,~]=MULTISCAN_JPDA0(XYZ,F,Q,H,R,X0,P0,Tracking_Scheme,JPDA_P,TN_H(end),...
    JPDA_multiscale,PD,S_limit,mui0,TPM,TPM_Option,H_TPM);

X_size=cellfun(@(x) size(x,2), XeT, 'UniformOutput', false);
Ff=cellfun(@(x,y,z) x(1):x(1)+y-1-z, Ff,X_size,Term_Con, 'ErrorHandler', @errorfun, ...
    'UniformOutput', false);
Ff_size=cellfun(@(x) size(x,2), Ff, 'UniformOutput', false);
XeT=cellfun(@(x,y) x(:,1:y),XeT,Ff_size, 'ErrorHandler', @errorfun, ...
    'UniformOutput', false);

[X_tr_j,Fr_tr_j,~,~]=Trajectory_Generator_Pruner_IMMJPDA(XeT,Ff,K,0);
[~,est_trk_j]=Track_Preprator(X_tr_j,Xdy,Fr_tr_j,Frdy,K);

figure,
    for nn=1:size(X_tr_j,2)
        plot(X_tr_j{nn}(1,:),X_tr_j{nn}(3,:),'s','Color',colorord(nn,:),'MarkerEdgeColor',colorord(nn,:),...
            'MarkerFaceColor',colorord(nn,:),'MarkerSize',4)
        hold on
    end
    axis([Surv_region(1,:) Surv_region(2,:)])


[dist_lospa1_LSTM(NoE,:),loce_lospa1_LSTM(NoE,:),carde_lospa1_LSTM(NoE,:),~,~] = perf_asses(xy_GT,est_trk_j(1:2:3,:,:),OSPA,'No');
clear Xtrg Ytrg XeT Ff est_trk_j X_tr_j Fr_tr_j JPDA_multiscale N_H tStart

%% 1 Frame, LSTM_HA
% JPDA Parameters
JPDA_multiscale=1; % Time-Frame windows
Tracking_Scheme='LSTM_HA';
[XeT,~,~,~,Ff,Term_Con,~]=MULTISCAN_JPDA0(XYZ,F,Q,H,R,X0,P0,Tracking_Scheme,JPDA_P,TN_H(end),...
    JPDA_multiscale,PD,S_limit,mui0,TPM,TPM_Option,H_TPM);

X_size=cellfun(@(x) size(x,2), XeT, 'UniformOutput', false);
Ff=cellfun(@(x,y,z) x(1):x(1)+y-1-z, Ff,X_size,Term_Con, 'ErrorHandler', @errorfun, ...
    'UniformOutput', false);
Ff_size=cellfun(@(x) size(x,2), Ff, 'UniformOutput', false);
XeT=cellfun(@(x,y) x(:,1:y),XeT,Ff_size, 'ErrorHandler', @errorfun, ...
    'UniformOutput', false);

[X_tr_j,Fr_tr_j,~,~]=Trajectory_Generator_Pruner_IMMJPDA(XeT,Ff,K,0);
[~,est_trk_j]=Track_Preprator(X_tr_j,Xdy,Fr_tr_j,Frdy,K);

figure,
    for nn=1:size(X_tr_j,2)
        plot(X_tr_j{nn}(1,:),X_tr_j{nn}(3,:),'s','Color',colorord(nn,:),'MarkerEdgeColor',colorord(nn,:),...
            'MarkerFaceColor',colorord(nn,:),'MarkerSize',4)
        hold on
    end
    axis([Surv_region(1,:) Surv_region(2,:)])


[dist_lospa1_LSTM_ha(NoE,:),loce_lospa1_LSTM_ha(NoE,:),carde_lospa1_LSTM_ha(NoE,:),~,~] = perf_asses(xy_GT,est_trk_j(1:2:3,:,:),OSPA,'No');
clear Xtrg Ytrg XeT Ff est_trk_j X_tr_j Fr_tr_j JPDA_multiscale N_H tStart

save('synthetic_results_new',...
    'dist_lospa1_jpda','loce_lospa1_jpda','carde_lospa1_jpda',...
    'dist_lospa1_jpda_ha','loce_lospa1_jpda_ha','carde_lospa1_jpda_ha',...
    'dist_lospa1_ha','loce_lospa1_ha','carde_lospa1_ha',...
    'dist_lospa1_LSTM_ha','loce_lospa1_LSTM_ha','carde_lospa1_LSTM_ha',...
    'dist_lospa1_LSTM','loce_lospa1_LSTM','carde_lospa1_LSTM')

end












