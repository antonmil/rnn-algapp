close all
clc

addpath('C:\gurobi651\win64\matlab')
gurobi_setup

K= 20;                                 %number of frames
T= 1;                                   %sampling period [s]
Num_Exp=10000; % number of experiments


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

% assert that data folder exists
if ~exist('Training_assignment_matrix', 'dir'), mkdir('Training_assignment_matrix'); end
if ~exist('Data', 'dir'), mkdir('Data'); end

tic
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
    
    %% 1 Frame, JPDA
    % JPDA Parameters
    JPDA_multiscale=1; % Time-Frame windows
    Tracking_Scheme='JPDA';
    mbest = 10;
%     Tracking_Scheme='JPDA_fst';
%     mbest = [];
    [XeT,~,~,~,Ff,Term_Con,~]=MULTISCAN_JPDA0_train(XYZ,F,Q,H,R,X0,P0,Tracking_Scheme,JPDA_P,mbest,...
        JPDA_multiscale,PD,S_limit,mui0,TPM,TPM_Option,H_TPM,NoE);
    
    clear Xtrg Ytrg XeT Ff est_trk_j X_tr_j Fr_tr_j JPDA_multiscale N_H tStart
    
    
end
toc

%% postprocess
% save to appropriate torch format
dirfiles = dir(['Training_assignment_matrix',filesep','Data*']);
nFiles = length(dirfiles);

N=5; M=N;
allQ = zeros(nFiles, N*N);
allMarginals = zeros(nFiles, N*N);
for f=1:nFiles
    dataFileName = fullfile('Training_assignment_matrix',dirfiles(f).name);
    da=load(dataFileName);
    oneQ=da.Assign_matrixx;
    mar = da.Fi_probabilty;
    allQ(f,:) = reshape(oneQ', 1, N*N);
    allMarginals(f,:) = reshape(mar', 1, N*N);
end

ttmode='train';
fname='LBP';
outf = fullfile('..','..','..','data',ttmode);
if ~exist(outf,'dir'), mkdir(outf); end
filename = sprintf('%s%s%s_N%d_M%d',outf,filesep,fname,N,M);
save([filename,'.mat'],'all*','-v7.3');
