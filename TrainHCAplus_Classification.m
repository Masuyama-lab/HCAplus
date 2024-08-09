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
function [Model, TF] = TrainHCAplus_Classification(Samples,net,Level,SampleLabels,maxLABEL)

% TRAINFTCA  Create the CA+ tree.

%%
TF = 0;
Model = [];
MaxLevel = net.MaxLevel;
[Dimension,NumSamples]=size(Samples);
if (Level>MaxLevel)    
    return;
end

%% Growing Process
NumSteps = net.Epochs*NumSamples;
Model = TrainCAplus_Classification(Samples,net,Level,NumSteps,SampleLabels,maxLABEL);

%% Expansion Process
Winners = Model.Winners;
Model.Means = Model.weight.';
NeuronsIndex = find(isfinite(Model.Means(1,:)));
NumNeurons = numel(NeuronsIndex);

%% PRUNE THE GRAPHS WITH ONLY 2 NEURONS. THIS IS TO SIMPLIFY THE HIERARCHY
if NumNeurons<=2
    Model=[];
    TF = 1;
    return;
else
    for NeuronIndex=NeuronsIndex
        ChildSamples = Samples(:,Winners==NeuronIndex);
        ChildSampleLabels = SampleLabels(Winners==NeuronIndex);
        Model.Child{NeuronIndex} = TrainHCAplus_Classification(ChildSamples,net,Level+1,ChildSampleLabels,maxLABEL);
    end
end

end