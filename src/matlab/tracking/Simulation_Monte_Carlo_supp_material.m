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

dist_lospa3_i = zeros(Num_Exp,K);
loce_lospa3_i= zeros(Num_Exp,K);
carde_lospa3_i= zeros(Num_Exp,K);

JPDA_Er_3F=cell(1,length(1:max(TN_H)));
NVHypo_3F=cell(Num_Exp,K);
TNH_3F=cell(Num_Exp,K);
TimeComp_3F=cell(Num_Exp,K);



for NoE=1:Num_Exp
    close all
    clc
    disp(['Experiment = ',num2str(NoE)])
% X= gen_state(F,Q0,X0,K);
% N_T=max(cellfun(@(x) size(x,2),X));
% [Z,iden]= gen_detection2(H,R,X,PD,Surv_region,lambda_c,1);
% XYZ=cellfun(@(x) x',Z,'UniformOutput', false);

load([pwd,'\Data\Experiment_',num2str(NoE)],'X','Z','iden','XYZ')

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

%% 3 Frames, probabilty comparison mbest Vs all assignments
% JPDA Parameters
JPDA_multiscale=3; % Time-Frame windows

[XeT,~,~,~,Ff,Term_Con,~,diff,NVHypo_3F(NoE,:),TNH_3F(NoE,:),TimeComp_3F(NoE,:)]=MULTISCAN_JPDA_Comparison_new(XYZ,F,Q,H,R,X0,P0,Tracking_Scheme,JPDA_P,TN_H,...
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
JPDA_Er_3F{1,kj}=[JPDA_Er_3F{1,kj};cell2mat(err_at_NH)];
end
clear  XeT Ff est_trk_j X_tr_j Fr_tr_j JPDA_multiscale N_H diff err_at_NH
save('synthetic_results_supp_material','JPDA_Er_3F','NVHypo_3F','TNH_3F'...
    ,'TimeComp_3F','dist_lospa3_i','loce_lospa3_i','carde_lospa3_i')
end

ttme=[];%ttme_niv=[];
for NoE=1:Num_Exp
for i=2:20
    for jh=1:size(TimeComp_3F{NoE,i}(2,:),2)
        if size(TimeComp_3F{NoE,i}{2,jh},2)==max(TN_H)
        ttme=[ttme;TimeComp_3F{NoE,i}{2,jh}];
        %ttme_niv=[ttme_niv;TimeComp_3F{NoE,i}{3,jh}];
        end
    end
end
end

close all
figure,plot(1:max(TN_H),mean(ttme,1),'-','Color',[0 0.5 0],'LineWidth',2);
hold on
plot(1:max(TN_H),mean(ttme_niv,1),'-m','LineWidth',2)
hold on
rectangle('Position',[0.5,0.01,99.5,1.98],'EdgeColor',[1 0.5 0],'LineStyle','--',...
                'LineWidth',2)

set(gca,'FontName','Times','FontSize',20),
set(gca, 'box', 'off')
set(gca, 'YColor', [0 0.5 0])
xlabel('Number of {\it m}-best solutions ')
ylabel('Processing time (sec.)','Color',[0 0.5 0]) % right y-axis


legend('Our {\it m}-best solver','Naive {\it m}-best solver','Location','NorthWest')
legend('boxoff')

print -dpsc 'mbest_time.eps'


close all
figure,plot(1:max(TN_H),mean(ttme,1),'-','Color',[0 0.5 0],'LineWidth',2);
hold on
plot(1:max(TN_H),mean(ttme_niv,1),'-m','LineWidth',2)
set(gca, 'box', 'off')
set(gca, 'YColor', [0 0.5 0])
hold on
rectangle('Position',[0.5,0.01,99.5,1.98],'EdgeColor',[1 0.5 0],'LineStyle','--',...
                'LineWidth',2)
axis([0 100 0 2])
set(gca,'YTick',0:0.5:2),

set(gca,'FontName','Times','FontSize',20),

xlabel('Number of {\it m}-best solutions ')
ylabel('Processing time (sec.)','Color',[0 0.5 0]) % right y-axis

legend('Our {\it m}-best solver','Naive {\it m}-best solver','Location','SouthEast')
legend('boxoff')
print -dpsc 'mbest_time_zoom.eps'

