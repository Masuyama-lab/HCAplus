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
function Model = TrainCAplus_Classification(Samples,net,Level,NumSteps,SampleLabels,maxLABEL)

Model.Level = Level;
Model.Samples=Samples;
Model.SampleLabels = SampleLabels;
DATA=Samples.'; 
Model.NumSteps=NumSteps;               % Total number of steps
Model.Winners=zeros(1,size(Samples,2));

numNodes = net.numNodes;               % the number of nodes
weight = net.weight;                   % node position
CountNode = net.CountNode;             % winner counter for each node

adaptiveSig = net.adaptiveSig;         % kernel bandwidth for CIM in each node
LabelCluster = net.LabelCluster;       % Cluster label for connected nodes
V_thres_ = net.V_thres_;               % similarlity thresholds
activeNodeIdx = net.activeNodeIdx;     % indexes of active nodes
CountLabel = net.CountLabel;           % a label counter
numSample = net.numSample;             % number of samples 
flag_set_lambda = net.flag_set_lambda; % a flag for setting lambda
numActiveNode = net.numActiveNode;     % number of active nodes
divMat = net.divMat;                   % a matrix for diversity via determinants
div_lambda = net.div_lambda;           % \lambda determined by diversity via determinants
sigma = net.sigma;                     % an estimated sigma for CIM

div_threshold = 1.0e-6;                % a threshold for diversity via determinants
n_init_data = 10;                      % number of signals for initialization of sigma

if size(weight) == 0
    CountLabel = zeros(1, maxLABEL);
end

if numSample == 0
    sigma = SigmaEstimationByNode(DATA(1:min(size(DATA,1),n_init_data),:),1:min(size(DATA,1),n_init_data));  
end

for sampleNum = 1:size(DATA,1)
    
    % Current data sample.
    input = DATA(sampleNum,:);
    label = SampleLabels(sampleNum, 1);
    numSample = numSample+1;
    
    if flag_set_lambda == false || numNodes < numActiveNode   
        % Generate 1st to bufferNode-th node from inputs.
        numNodes = numNodes + 1;
        weight(numNodes,:) = input;
        activeNodeIdx = updateActiveNode(activeNodeIdx, numNodes);
        CountNode(numNodes) = 1;
        
        if length(adaptiveSig)~= 0
            adaptiveSig(numNodes) = adaptiveSig(1);
        end
        CountLabel(numNodes,label) = 1;
        Model.Winners(sampleNum)=numNodes;
      
        if size(weight ,1) >= n_init_data && flag_set_lambda == false
            Corr = 1-CIM(weight(numNodes,:),weight,median(sigma));
            divMat(numNodes,1:numNodes) = Corr'; 
            divMat(1:numNodes,numNodes) = Corr;  
            Div = det(exp(divMat));
            
            if  Div < div_threshold && size(weight ,1) >= n_init_data
                numActiveNode = numNodes;
                div_lambda = numActiveNode*2;
            end
        end
        
        % Calculate the initial similarlity threshold to the initial nodes.
        if numNodes == numActiveNode
            flag_set_lambda = true;
            numAN = size(activeNodeIdx,2);
            initSig = SigmaEstimationByNode(weight, activeNodeIdx(1:min(numAN,numActiveNode)));
            adaptiveSig = repmat(initSig,1,numNodes); % Assign the same initSig to the all nodes.           
            tmpTh = zeros(1,numActiveNode);
            for k = 1:numActiveNode
                tmpCIMs1 = CIM(weight(k,:), weight, mean(adaptiveSig));
                [~, s1] = min(tmpCIMs1);
                tmpCIMs1(s1) = inf; % Remove CIM between weight(k,:) and weight(k,:).
                tmpTh(k) = min(tmpCIMs1);
            end
            V_thres_ = mean(tmpTh);
        end
    else
               
        % Calculate CIM based on global mean adaptiveSig.
        globalCIM = CIM(input, weight, mean(adaptiveSig));
        
        % Set CIM state between the local winner nodes and the input for Vigilance Test.
        [Vs1, s1] = min(globalCIM);
        globalCIM(s1) = inf;
        [Vs2, s2] = min(globalCIM);
        
        if V_thres_ < Vs1 || numNodes < numActiveNode% Case 1 i.e., V < CIM_k1
            % Add Node
            numNodes = numNodes + 1;
            weight(numNodes,:) = input;
            activeNodeIdx = updateActiveNode(activeNodeIdx, numNodes);
            CountNode(numNodes) = 1;

            CountLabel(numNodes,label) = 1;
            Model.Winners(sampleNum)=numNodes;
            
            numAN = size(activeNodeIdx,2);
            adaptiveSig(numNodes) = SigmaEstimationByNode(weight, activeNodeIdx(1:min(numAN,numActiveNode)));
 
        else % Case 2 i.e., V >= CIM_k1
                      
            % Update s1 weight
            CountNode(s1) = CountNode(s1) + 1;
            weight(s1,:) = weight(s1,:) + (1/CountNode(s1)) * (input - weight(s1,:));
            activeNodeIdx = updateActiveNode(activeNodeIdx, s1);
            
           
            Model.Winners(sampleNum)=s1;
            CountLabel(s1,label) = CountLabel(s1, label) + 1;

            if V_thres_ >= Vs2 % Case 3 i.e., V >= CIM_k2
                % Update weight of s2 node.
                weight(s2,:) = weight(s2,:) + (1/(100*CountNode(s2))) * (input - weight(s2,:));
            end  
                   
        end % if minCIM < Lcim_s1 % Case 1 i.e., V < CIM_k1
    end % if size(weight,1) < 2
    
end % for sampleNum = 1:size(DATA,1)

if length(adaptiveSig) == 0
   numActiveNode = numNodes;
   div_lambda = numActiveNode*2;
   numAN = size(activeNodeIdx,2);
   initSig = SigmaEstimationByNode(weight, activeNodeIdx(1:min(numAN,numActiveNode)));
   adaptiveSig = repmat(initSig,1,numNodes); % Assign the same initSig to the all nodes.
          
   tmpTh = zeros(1,numActiveNode);
   for k = 1:numActiveNode
       tmpCIMs1 = CIM(weight(k,:), weight, mean(adaptiveSig));
       [~, s1] = min(tmpCIMs1);
       tmpCIMs1(s1) = inf; % Remove CIM between weight(k,:) and weight(k,:).
       tmpTh(k) = min(tmpCIMs1);
   end
   V_thres_ = mean(tmpTh);
end

% connection = graph(edge~= 0);
% LabelCluster = conncomp(connection);
if isempty(LabelCluster)
  LabelCluster = 0;
end

Model.numNodes = numNodes;
Model.weight = weight;
Model.CountNode = CountNode;
Model.adaptiveSig = adaptiveSig;
Model.LabelCluster = LabelCluster;
Model.V_thres_ = V_thres_;
Model.activeNodeIdx = activeNodeIdx;
Model.CountLabel = CountLabel;
Model.CL = CountLabel';
Model.numSample = numSample;
Model.flag_set_lambda = flag_set_lambda;
Model.numActiveNode = numActiveNode;
Model.divMat = divMat;
Model.div_lambda = div_lambda;
Model.sigma = sigma;
end




% Compute an initial kernel bandwidth for CIM based on data points.
function estSig = SigmaEstimationByNode(weight, activeNodeIdx)

exNodes = weight(activeNodeIdx,:);
% Add a small value for handling categorical data.
qStd = std(exNodes);
qStd(qStd==0) = 1.0E-6;

% normal reference rule-of-thumb
% https://www.sciencedirect.com/science/article/abs/pii/S0167715212002921
[n,d] = size(exNodes);
estSig = median( ((4/(2+d))^(1/(4+d))) * qStd * n^(-1/(4+d)) );

end
 

% Correntropy induced Metric
function cim = CIM(X,Y,sig)
% X : 1 x n
% Y : m x n
cim = sqrt(1 - mean(exp(-(X-Y).^2/(2*sig^2)), 2))';
end


function activeNodeIdx = updateActiveNode(activeNodeIdx, winnerIdx)
%activeNodeIdx
activeNodeIdx(activeNodeIdx == winnerIdx)= [];
activeNodeIdx = [winnerIdx,activeNodeIdx];
end

