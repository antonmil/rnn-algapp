close all
clear all
clc

K= 20;                                 %number of frames
T= 1;                                   %sampling period [s]

u_image=30;v_image=30;
Surv_region=[1 u_image; 1 v_image];     %survillance area
X0=[5 1 13 0.2;5 1 15 0;5 1 17 -0.2];


F11=[1 T;0 1];
F(:,:,1)=blkdiag(F11,F11); % The transition matrix for the dynamic model 1
q1=0.0001; % The standard deviation of the process noise for the dynamic model 1
Q11=q1*[T^3/3 T^2/2;T^2/2 T];
Q(:,:,1)=blkdiag(Q11,Q11); % The process covariance matrix for the dynamic model 1
P0=Q(:,:,1);
N_T=max(cellfun(@(x) size(x,2),X));

q2=0.01;
Q22=q2*[T^3/3 T^2/2;T^2/2 T];
Q(:,:,1)=blkdiag(Q22,Q22); % The process covariance matrix for the dynamic model 1


% Measurement Matrices
lambda_c=3;% Mean number of clutter per frame

H=[1 0 0 0;0 0 1 0]; % % Measurement matrix
R=[.05 0;0 0.05]; % Measurement covariance matrix
PD=0.7; % Probabilty of detection


% [Z,iden]= gen_detection(H,R,X,PD,Surv_region,lambda_c);


JPDA_P=[lambda_c/(u_image*v_image),15];%Beta=2/(u_image*v_image);Gate=30;
S_limit=50;
 

% PD Parameters
PD_Option='Constant'; % PD_Option='State-Dependent';
H_PD=[1 0 0 0;0 0 1 0];

% IMM_Parameters
mui0=1; % The initial values for the IMM probability weights
TPM_Option='Constant'; % TPM_Option='State-Dependent';
TPM=1; % TPM{2,2}=1;TPM{2,1}=2*eye(2,2);TPM{1,2}=1;TPM{1,1}=0.5*eye(2,2);
H_TPM=[0 1 0 0;0 0 0 1];

Initiation='TotalBased'; % The parameter for initiation

Tracking_Scheme='JPDA';



X= gen_state(F,Q,X0,K);
[Z,iden]= gen_detection2(H,R,X,PD,Surv_region,lambda_c,1);

XYZ=cellfun(@(x) x',Z,'UniformOutput', false);

 % Threshold for considering maximum number of Hypotheses
JPDA_multiscale=3; % Time-Frame windows
cnt=0;
for N_H=1:10:100
cnt=cnt+1;
[~,~,~,~,~,~,~,diff,NVHypo,TNH]=MULTISCAN_JPDA_Comparison(XYZ,F,Q,H,R,X0,P0,Tracking_Scheme,JPDA_P,N_H,...
    JPDA_multiscale,PD,S_limit,mui0,TPM,TPM_Option,H_TPM);
diff(cellfun(@isempty,diff))=[];
Diff=cell2mat(diff);
Meann(cnt)=mean(Diff);
Minn(cnt)=min(Diff);
Maxx(cnt)=max(Diff);
end