#include<math.h>
#include<iostream>
#include <fstream>


#include"Eigen/Dense"
#include"Eigen/StdVector"

#include "helper_funcs.h"
#include "layer.cpp"



using namespace Eigen;

class MLP2 {
  private:
    int num_hidden_layers = 2;
    MatrixXd inputs;
    MatrixXd teacher_output;
    MatrixXd network_output;
    std::vector<MLPLayer*> all_layers;
    std::vector<int> *num_neurons;
    std::vector<double> *eta_per_layer;// learning rates

  public:
    MLP2(MatrixXd& training_inputs, MatrixXd& teacher_outputs,std::vector<int> *num_neuron,std::vector<double> *etas) {
        this->inputs = training_inputs;
        this->teacher_output = teacher_outputs;
        this->num_hidden_layers=num_neuron->size()-2; // minus input and output
        this->num_neurons=num_neuron;
        this->eta_per_layer=etas;
        init();
    }

    MLP2(int num_hidden_layers = 2) {
        this->num_hidden_layers=num_hidden_layers;
        init();
    }
    void init() {
        if(inputs.rows() == 0 || teacher_output.rows()==0) {
            std::cerr << "Training inputs and/or teacher outputs is empty"<<std::endl;
            exit(255);
        }

        // setup input layer
        MLPLayer *input = (new MLPLayer())
                          ->setActivation(IDENTITY)
                          ->setType(INPUT)
                          ->setBatch_size(inputs.rows())
                          ->setNum_neurons(num_neurons->at(0)+1) // added 1 for bias weight
                          ->setNum_outputs(num_neurons->at(1));
        input->init();
        all_layers.push_back(input);


        // instantiate and setup hidden layers
        for (int i = 1; i < num_hidden_layers+1; ++i) {
            MLPLayer *mlpl = (new MLPLayer())
                             ->setActivation(LOGISTIC)
                             ->setType(HIDDEN)
                             ->setBatch_size(inputs.rows())
                             ->setNum_neurons(num_neurons->at(i)+1) // added 1 for bias weight
                             ->setNum_outputs(num_neurons->at(i+1));
            mlpl->init();
            all_layers.push_back(mlpl);
        }

        // setup output layer
        MLPLayer *output = (new MLPLayer())
                           ->setActivation(LOGISTIC)
                           ->setType(OUTPUT)
                           ->setBatch_size(inputs.rows())
                           ->setNum_neurons(num_neurons->at(num_neurons->size()-1))
                           ->setNum_outputs(teacher_output.cols());
        output->init();
        all_layers.push_back(output);
    }

    Eigen::MatrixXd forward_propagate(MatrixXd input_data) {
        //append bias to input data
        MatrixXd input(input_data);
        input.conservativeResize(input.rows(), input.cols()+1);
        MatrixXd bias;
        bias.setOnes(input.rows(), 1);
        input.col(input.cols()-1) = bias;
//        delete all_layers.at(0)->getZ();
        all_layers.at(0)->setZ(input);

        // forward propagate hidden layers
        for (int layer = 0; layer < all_layers.size()-1; ++layer) {
            MatrixXd temp = all_layers.at(layer)->forward_propagate();
            all_layers.at(layer+1)->setS(temp);
        }
        network_output = all_layers.at(all_layers.size()-1)->forward_propagate();
        return network_output;
    }

    std::vector<MLPLayer*> getLayers() const {
        return all_layers;
    }

    void backward_propagate(MatrixXd network_y, MatrixXd teacher_y) {
        MatrixXd temp  = (network_y - teacher_y).transpose();
        all_layers.at(all_layers.size()-1)->setD(temp);

        for (int lay_idx = all_layers.size()-2; lay_idx > 0; --lay_idx) {
            MLPLayer *curr_layer = all_layers.at(lay_idx);
            MatrixXd w = curr_layer->getW();
            MatrixXd W_nobias = w.topRows(w.rows()-1);

            MatrixXd next_D = all_layers.at(lay_idx+1)->getD();
            MatrixXd temp = (W_nobias * next_D).cwiseProduct(curr_layer->getF());
            curr_layer->setD(temp);
        }
    }

    void update_weights() {
        #pragma omp parallel for
        for (int idx = 0; idx < all_layers.size()-1; ++idx) {
            MatrixXd next_D = all_layers.at(idx+1)->getD();
            MatrixXd curr_Z = all_layers.at(idx)->getZ();
            MatrixXd W_gradient = -eta_per_layer->at(idx) * (next_D*curr_Z).transpose();
            all_layers.at(idx)->addGradToW(W_gradient);
        }
    }

    double evaluate(MatrixXd& test_input, MatrixXd& test_output, int iters = 10000) {
        std::fstream output_file("error.dat", std::ios_base::out);
        if(!output_file.is_open()) {
            std::cerr << "Not open!! "<< std::endl;
            return 0.0;
        }
        for (int iter = 0; iter < iters; ++iter) { // repeat training

            for (int row_idx = 0; row_idx < inputs.rows(); ++row_idx) { //train
                forward_propagate(inputs.row(row_idx));
                backward_propagate(network_output, teacher_output.block(row_idx,0,1,teacher_output.cols()) );
                update_weights();
            }

            //calculate training error
            Eigen::MatrixXd out = forward_propagate(inputs);
            double err = ((teacher_output) - (out)).squaredNorm() / teacher_output.rows();
            std::cout << "[train] Sum of squared errors: "<< err <<std::endl;
            output_file <<iter<< "\t"<< err << "\t";
//            delete out;
            //calculate training error
            out = forward_propagate(test_input);
            err = ((test_output) - (out)).squaredNorm()/ test_output.rows();
            std::cout << "[test] Sum of squared errors: "<< err <<std::endl;
            output_file << err <<std::endl;


        }
        output_file.close();
    }
};

