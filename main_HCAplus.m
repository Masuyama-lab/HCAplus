% 
% (c) 2024 Naoki Masuyama
% 
% These are the codes of Hierarchical CIM-based ART+ (HCA+)
% proposed in K. Tashiro, N. Masuyama, and Y. Nojima, 
% "A growing hierarchical clustering algorithm via parameter-free adaptive resonance theory," 
% in Proc. of 2024 International Joint Conference on Neural Networks (IJCNN), 2024.
% 
% Please contact "masuyama@omu.ac.jp" if you have any problems.
%    

clear all; close all; % clc; clear all; dbstop if error; close all;

% Experimental Conditions==================================================
data_name= 'Wine';                  % Dataname {Iris, Sonar, Wine}
Epoch = 1;                          % Number of epochs for data input count
maxlevel = 10;                      % StopLevel
%  ========================================================================

% Data preparation ------------------------------------------------
tmpData = load(strcat(string(data_name),'.mat'));
IMAGES = tmpData.data;
LABELS = tmpData.target;
maxLABEL = max(LABELS);

% Randamization
ran = randperm(size(IMAGES,1));
ranData = IMAGES(ran,:)';
ranLabel = LABELS(ran);

% ----------------------------------------------------------------

% Parameters of HCAplus ==========================================
HCAplusNet.numNodes = 0;           % the number of nodes
HCAplusNet.weight = [];            % node position
HCAplusNet.CountNode = [];         % winner counter for each node
HCAplusNet.adaptiveSig = [];       % kernel bandwidth for CIM in each node
HCAplusNet.V_thres_ = [];          % similarlity thresholds
HCAplusNet.activeNodeIdx = [];     % nodes for SigmaEstimation
HCAplusNet.numSample = 0;          % counter for input sample
HCAplusNet.flag_set_lambda = false;% flag for a calculation of lambda
HCAplusNet.numActiveNode = inf;    % the number of active nodes
HCAplusNet.div_lambda = inf;       % numActiveNodes * 2
HCAplusNet.divMat = [];            % matrix for a pairwise similarity
HCAplusNet.sigma = [];             % sigma defined by initial nodes
HCAplusNet.CountLabel = []; 
HCAplusNet.LabelCluster = [];
HCAplusNet.MaxLevel = maxlevel; 
HCAplusNet.Epochs  =Epoch; 
% ================================================================
time_HCAplus = 0;

% Training -------------------------------------------------------
Level = 1;
tic
[HCAplusNet,TF] = TrainHCAplus_Classification(ranData,HCAplusNet,Level,ranLabel,maxLABEL);
time_HCAplus = time_HCAplus + toc/(Epoch*size(ranData',1));
%-----------------------------------------------------------------  

if ~(HCAplusNet.numNodes == 0)
     %get leaves--------------------------------------------------
     [LEAVESnet,MaxLevel] = HCAplus_GetLEAVESnet_Classification(HCAplusNet,0);
     LEAVESnet.weight = LEAVESnet.Means;          
     LEAVESnet.CountLabel = LEAVESnet.CL;
     %------------------------------------------------------------

     % Evaluation ------------------------------------------------
     [~, HCAplus_normMI,~,~, HCAplus_ARI] = HCAplus_Evaluation(ranData', ranLabel, LEAVESnet);
     num_nodes_HCAplus = CountNumNodes(HCAplusNet); %Counting the number of all nodes
     %------------------------------------------------------------                         
end   

% Results
disp('Results of HCA+:');
disp(['Number of Leaf Nodes: ', num2str(LEAVESnet.numNodes)]);
disp(['Number of Nodes: ',num2str(num_nodes_HCAplus)]);
disp(['Number of Layer: ',num2str(MaxLevel)]);
disp(['NMI: ',num2str(HCAplus_normMI)]);
disp(['ARI: ',num2str(HCAplus_ARI)]);
disp(['Processing Time: ',num2str(time_HCAplus)]);

