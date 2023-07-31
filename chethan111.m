% Load and preprocess the data
data = imageDatastore('C:\Users\chetan\Documents\MATLAB\lavanya\model training\retinopathi', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
data.ReadFcn = @preprocessImages; % Define your own preprocessing function

% Split the dataset into training, validation, and testing sets
[trainData, valData, testData] = splitImageDatastore(data, 1000, 250, 250);

% Load pretrained AlexNet
net = alexnet('Weights', 'imagenet');
layers = net.Layers;

% Modify the last layers for the specific classification task
numClasses = numel(categories(trainData.Labels));
layers(end-2) = fullyConnectedLayer(numClasses);
layers(end) = classificationLayer;

% Set training options
options = trainingOptions('sgdm', 'MaxEpochs', 10, 'MiniBatchSize', 32, 'InitialLearnRate', 0.001, 'Plots', 'training-progress', 'ValidationData', valData);

% Train the network
trainedNet = trainNetwork(trainData, layers, options);

% Evaluate the network on the test set
predictions = classify(trainedNet, testData);
accuracy = mean(predictions == testData.Labels);

fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Helper function to preprocess the images
function img = preprocessImages(filename)
    % Read the image
    img = imread(filename);
    
    % Preprocess the image (e.g., resize, normalize, etc.)
    img = imresize(img, [227, 227]); % Adjust the size as per your requirement
    img = im2double(img);
end

% Helper function to split the image datastore into train, validation, and test sets
function [trainData, valData, testData] = splitImageDatastore(data, numTrain, numVal, numTest)
    % Shuffle the image datastore
    data = shuffle(data);
    
    % Split the shuffled datastore based on the specified number of images
    trainData = subset(data, 1:numTrain);
    valData = subset(data, numTrain+1:numTrain+numVal);
    testData = subset(data, numTrain+numVal+1:numTrain+numVal+numTest);
end
