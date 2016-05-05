close all
clear all
clc

K= 20;                                 %number of frames
T= 1;                                   %sampling period [s]

u_image=30;v_image=30;
Surv_region=[1 u_image; 1 v_image];     %survillance area
X0=[5 1 13 0.2;5 1 15 0;5 1 17 -0.2;];


F11=[1 T;0 1];
F(:,:,1)=blkdiag(F11,F11); % The transition matrix for the dynamic model 1
q1=0.0001; % The standard deviation of the process noise for the dynamic model 1
Q11=q1*[T^3/3 T^2/2;T^2/2 T];
Q(:,:,1)=blkdiag(Q11,Q11); % The process covariance matrix for the dynamic model 1
P0=Q(:,:,1);
X= gen_state(F,Q,X0,K);
N_T=max(cellfun(@(x) size(x,2),X));
figure,
colorord=[1 0 0;0 1 0;0 0 1; 1 1 0;1 0 1;0 1 1];

for k=1:K
    for nn=1:size(X{k},2)
        plot(X{k}(1,nn),X{k}(3,nn),'.','Color',colorord(nn,:))
        hold on
    end
    axis([Surv_region(1,:) Surv_region(2,:)])
    pause(0.05)
end
q2=0.01;
Q22=q2*[T^3/3 T^2/2;T^2/2 T];
Q(:,:,1)=blkdiag(Q22,Q22); % The process covariance matrix for the dynamic model 1


% Measurement Matrices
lambda_c=5;% Mean number of clutter per frame

H=[1 0 0 0;0 0 1 0]; % % Measurement matrix
R=[.05 0;0 0.05]; % Measurement covariance matrix
PD=0.7; % Probabilty of detection


% [Z,iden]= gen_detection(H,R,X,PD,Surv_region,lambda_c);
[Z,iden]= gen_detection2(H,R,X,PD,Surv_region,lambda_c,1);
figure,

for k=1:K
    ix0=find(iden{k}==0);
    plot(Z{k}(1,ix0),Z{k}(2,ix0),'*k')
    for nn=1:size(X{k},2)
        ix=find(iden{k}==nn);
        if ~isempty(ix)
            plot(Z{k}(1,ix),Z{k}(2,ix),'*','Color',colorord(iden{k}(ix),:))
            hold on
        end
    end

    axis([Surv_region(1,:) Surv_region(2,:)])
    pause(0.05)
end
% for nn=1:size(X0,1)
%     Xtrg=zeros(1,K);
%     Ytrg=zeros(1,K);
%     for k=1:K
%         Xtrg(k)=X{k}(1,nn);
%         Ytrg(k)=X{k}(3,nn);
%     end
%     plot(Xtrg,Ytrg,'-','Color',colorord(nn,:))
%     hold on
%     axis([Surv_region(1,:) Surv_region(2,:)])
% end

% figure, 
% 
% for k=1:K
%     ix0=find(iden{k}==0);
%     plot3(Z{k}(1,ix0),Z{k}(2,ix0),k*ones(1,length(ix0)),'*k')
%     for nn=1:size(X{k},2)
%         ix=find(iden{k}==nn);
%         if ~isempty(ix)
%             plot3(Z{k}(1,ix),Z{k}(2,ix),k*ones(1,length(ix)),'.','Color',colorord(iden{k}(ix),:))
%             hold on
%         end
%     end
%     axis([Surv_region(1,:) Surv_region(2,:) 1 K])
%     pause(0.05)
% end

% writerObj =  VideoWriter('Detections.avi','MPEG-4');
% writerObj.FrameRate = 5;
% open(writerObj);
% 
% scrsz = get(0,'ScreenSize');
%   
%  for k=1:K
%      fig4b=figure('Position',[scrsz(1) scrsz(2) scrsz(3) scrsz(4) ]);
%      set(fig4b, 'Color',[0 0 0])
%      ix0=find(iden{k}==0);
%      plot(Z{k}(1,ix0),Z{k}(2,ix0),'*k')
%      hold on
%      for nn=1:size(X{k},2)
%          ix=find(iden{k}==nn);
%          if ~isempty(ix)
%              plot(Z{k}(1,ix),Z{k}(2,ix),'o','Color',colorord(iden{k}(ix),:))
%              hold on
%          end
%      end
%     axis([Surv_region(1,:) Surv_region(2,:)])
%     frrr = getframe(fig4b);
%     writeVideo(writerObj,frrr);
%     close(fig4b);
%  end
 
%  close(writerObj);

XYZ=cellfun(@(x) x',Z,'UniformOutput', false);

JPDA_P=[lambda_c/(u_image*v_image),20];%Beta=2/(u_image*v_image);Gate=30;
S_limit=100;
 

% PD Parameters
PD_Option='Constant'; % PD_Option='State-Dependent';
H_PD=[1 0 0 0;0 0 1 0];

% IMM_Parameters
mui0=1; % The initial values for the IMM probability weights
TPM_Option='Constant'; % TPM_Option='State-Dependent';
TPM=1; % TPM{2,2}=1;TPM{2,1}=2*eye(2,2);TPM{1,2}=1;TPM{1,1}=0.5*eye(2,2);
H_TPM=[0 1 0 0;0 0 0 1];

Initiation='TotalBased'; % The parameter for initiation
%% T Frames, M-Best assignment
% JPDA Parameters
Tracking_Scheme='JPDA';
JPDA_multiscale=3; % Time-Frame windows
N_H=50; % Threshold for considering maximum number of Hypotheses

tStart = tic; 
[XeT,PeT,Xe,Pe,Ff,Term_Con,mui]=MULTISCAN_JPDA0(XYZ,F,Q,H,R,X0,P0,Tracking_Scheme,JPDA_P,N_H,...
    JPDA_multiscale,PD,S_limit,mui0,TPM,TPM_Option,H_TPM);
tElapsed = toc(tStart);

X_size=cellfun(@(x) size(x,2), XeT, 'UniformOutput', false);
Ff=cellfun(@(x,y,z) x(1):x(1)+y-1-z, Ff,X_size,Term_Con, 'ErrorHandler', @errorfun, ...
    'UniformOutput', false);
Ff_size=cellfun(@(x) size(x,2), Ff, 'UniformOutput', false);
XeT=cellfun(@(x,y) x(:,1:y),XeT,Ff_size, 'ErrorHandler', @errorfun, ...
    'UniformOutput', false);


figure,
 for k=1:K
    for n=1:size(XeT,2)
     idx=find(Ff{1,n}==k);
     if ~isempty(idx)
          plot(XeT{1,n}(1,idx),XeT{1,n}(3,idx),'o','Color',colorord(n,:))
              hold on
     end
    end
axis([Surv_region(1,:) Surv_region(2,:)])
    pause(0.05)
 end
for nn=1:size(X0,1)
        Xtrg=zeros(1,K);
        Ytrg=zeros(1,K);
    for k=1:K
        Xtrg(k)=X{k}(1,nn);
        Ytrg(k)=X{k}(3,nn);
    end
    plot(Xtrg,Ytrg,'-','Color',colorord(nn,:))
    hold on
    axis([Surv_region(1,:) Surv_region(2,:)])
end
title(['Hypothesis # = ',num2str(N_H),', Frames # = ',num2str(JPDA_multiscale)])

