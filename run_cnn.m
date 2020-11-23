clear all
close all
clc

%% This function simply runs the CNN algorithm with the given architecture.

load('dataset.mat');

input_data = traindat(1,:);

% Extract the individual red, green, and blue color channels.
redChannel = rot90(reshape(input_data(1:1024),32,32),3);
greenChannel = rot90(reshape(input_data(1024+1:1024*2),32,32),3);
blueChannel = rot90(reshape(input_data(1024*2+1:1024*3),32,32),3);
% Recombine separate color channels into an RGB image.
rgbImage = cat(3, redChannel, greenChannel, blueChannel);
% % Show the image
% imagesc(rgbImage)

%% Reduce the dataset.

% Note that labels are in the interval [0,9]. For the sake of matlab
%   implementation, we need them to be in the interval [1,10].

trainlbl = trainlbl + 1;
testlbl = testlbl + 1;

% Reduce dataset if requested
reduction_ratio = 0.1;

single_class = 'true';
single_class_id = [1 3 5 7 9]; % 5 7 9

if ~(reduction_ratio >= 1)

    new_trainlbl_indices = [];
    new_testlbl_indices = [];

    for i_class = 1:10
        
        if and( strcmp(single_class,'true') , isempty(find(single_class_id == i_class)) )
            % Do nothing
        else

            % For train data
            train_class_indices = find(trainlbl == i_class);

            new_len = ceil(length(train_class_indices)*reduction_ratio);

            new_rand_incides = randperm(length(train_class_indices),new_len);
            new_train_class_indices = train_class_indices(new_rand_incides);

            new_trainlbl_indices = [new_trainlbl_indices ; new_train_class_indices(:)];


            % For test data
            test_class_indices = find(testlbl == i_class);

            new_len = ceil(length(test_class_indices)*reduction_ratio);

            new_rand_incides = randperm(length(test_class_indices),new_len);
            new_test_class_indices = test_class_indices(new_rand_incides);

            new_testlbl_indices = [new_testlbl_indices ; new_test_class_indices(:)];
        
        end


    end
    
    new_trainlbl_indices = new_trainlbl_indices(randperm(length(new_trainlbl_indices)));
    new_testlbl_indices = new_testlbl_indices(randperm(length(new_testlbl_indices)));
    
    trainlbl = trainlbl(new_trainlbl_indices);
    testlbl = testlbl(new_testlbl_indices);
    
    traindat = traindat(new_trainlbl_indices,:);
    testdat = testdat(new_testlbl_indices,:);

end

%% Prepare dataset for processing


% Prepare train, validation and test datasets.

% !!!! This part must be reorganized !!!!

train_data = traindat; %( 1:40000 , : );
train_label = trainlbl; %( 1:40000 , : );

validation_data = traindat; %( 40001 : end , : );
validation_label = trainlbl; %( 40001 : end , : );

validation_section = 1;

test_data = testdat;
test_label = testlbl;



%% Prepare the inputs of the algorithm

% Convolutional Layers' Settings
input_struct.conv_input_channel = 3;
input_struct.conv_input_size = size(redChannel);
input_struct.conv_layer_content = [16 20 20];
input_struct.conv_learning_rate = [0.1 0.1 0.1];
input_struct.conv_zero_padding_size = [2 2 2];
input_struct.conv_kernel_size = {[5,5] , ...
                                 [5,5] , ...
                                 [5,5]};
input_struct.conv_kernel_init = {{'normal', 0 , 0.01} , ... 
                                 {'normal', 0 , 0.01} , ...
                                 {'normal', 0 , 0.01}};
input_struct.conv_pooling_type = {'max' , ...
                                  'max' , ...
                                  'max'};
input_struct.conv_pooling_size = {[2,2] , ...
                                  [2,2] , ...
                                  [2,2]};
input_struct.conv_bias_type = {'untied' , ...
                               'untied' , ...
                               'untied'};
input_struct.conv_activation = {'ReLU' , ...
                                'ReLU' , ...
                                'ReLU'};
                            
% Fully Connected Layers' Settings
input_struct.fulconn_output_neuron_count = 10; % Fixed
input_struct.fulconn_hidden_layer_neuron_content = []; 
input_struct.fulconn_learning_rate = [0.1];
input_struct.fulconn_weight_init = {{'uniform',-0.1 , 0.1}};
input_struct.fulconn_activation = {'softmax'}; 
input_struct.fulconn_momentum_coef = 0.9;
input_struct.fulconn_weight_decay = 0.0001;

% Network's General Settings
input_struct.epoch_num = 10;
input_struct.minibatch_size = 20;
input_struct.train_data = train_data;
input_struct.train_label = train_label;
input_struct.validation_data = validation_data;
input_struct.validation_label = validation_label;
input_struct.test_data = test_data;
input_struct.test_label = test_label;
input_struct.validation_section = validation_section;


% Run the algorithm
[output_struct] = implement_cnn_algorithm(input_struct);

% Gather the outputs

% Error metrics (mean cross entropy error and mean classification error)
train_mcee_per_epoch = output_struct.train_mcee_per_epoch;  
train_mce_per_epoch = output_struct.train_mce_per_epoch;       
validation_mcee_per_epoch = output_struct.validation_mcee_per_epoch;  
validation_mce_per_epoch = output_struct.validation_mce_per_epoch;       
test_mcee_per_epoch = output_struct.test_mcee_per_epoch;  
test_mce_per_epoch = output_struct.test_mce_per_epoch;      
% Resultant weight per epoch
conv_weight_per_epoch = output_struct.conv_weight_per_epoch; 
conv_bias_per_epoch = output_struct.conv_bias_per_epoch; 
fulconn_weight_per_epoch = output_struct.fulconn_weight_per_epoch; 
% Total elapsed time
total_elapsed_time = output_struct.total_elapsed_time;


%% Plot the results

h = figure

subplot(1,2,1)
plot(train_mcee_per_epoch)
hold on
% plot(validation_mcee_per_epoch)
plot(test_mcee_per_epoch)
legend('Train MCCE','Test MCCE');
% legend('Train MCCE','Validation MCCE','Test MCCE');
xlabel('epoch')
ylabel('mcce')

subplot(1,2,2)
plot(train_mce_per_epoch)
hold on
% plot(validation_mce_per_epoch)
plot(test_mce_per_epoch)
legend('Train MCE','Test MCE');
% legend('Train MCE','Validation MCE','Test MCE');
xlabel('epoch')
ylabel('mce')

disp(['Total Elapsed Time: ' num2str(total_elapsed_time)])


save('all_output_data.mat');