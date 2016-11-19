#include "Eigen/Dense"
enum TRANS_FUN {
  LOGISTIC,
  TANH,
  IDENTITY
};
enum LAYER_TYPE {
  INPUT,
  HIDDEN,
  OUTPUT
};

class MLPLayer {
  friend class MLP2;
private:
  LAYER_TYPE type;
  TRANS_FUN activation;
  int num_neurons;
  int num_outputs;
  int batch_size;
  Eigen::MatrixXd *Z, // outputs
  *W, // outgoing weights
  *S, // inputs
  *D, // deltas for this layer
  *F; // derivatives of the activation function

public:
  MLPLayer() {

  }

  MLPLayer(int num_input_neurons,int batch_size,
           LAYER_TYPE type=HIDDEN,TRANS_FUN activation=LOGISTIC) {
    this->type = type;
    this->activation = activation;
    this->num_neurons=num_input_neurons;
    this->batch_size = batch_size;

  }
  Eigen::MatrixXd * forward_propagate() {
    if (type==INPUT) {
        Eigen::MatrixXd *res= new Eigen::MatrixXd((*Z)*(*W));
        return res;
      }

    Z = activate(S);

    if (type==OUTPUT) {
        return Z;
      }

    // hidden layer --> add bias
    Z->conservativeResize(Z->rows(), Z->cols()+1);
    Eigen::MatrixXd bias;
    bias.setOnes(Z->rows(), 1);
    Z->col(Z->cols()-1) = bias;

    // calc F and transpose
    delete F;
    F = new Eigen::MatrixXd(activate(S, true)->transpose());

    // return ZxW
    return new Eigen::MatrixXd((*Z)*(*W));
  }
  Eigen::MatrixXd * activate(Eigen::MatrixXd * mat, bool deriv = false) {
    if(!deriv)
      switch (activation) {
        case TANH:
          return new Eigen::MatrixXd(mat->array().tanh());
          break;
        case LOGISTIC:
          return new Eigen::MatrixXd(mat->unaryExpr(&logistic));
          break;
        case IDENTITY:
          return new Eigen::MatrixXd (*mat);
          break;
        default:
          break;
        }
    else
      switch (activation) {
        case TANH:
          return new Eigen::MatrixXd(mat->unaryExpr(&tanh_deriv));
          break;
        case LOGISTIC:
          return new Eigen::MatrixXd(mat->unaryExpr(&logistic_deriv));
          break;
        case IDENTITY:
          return new Eigen::MatrixXd (*mat);
          break;
        default:
          break;
        }
  }

  TRANS_FUN getActivation() const {
    return activation;
  }

  MLPLayer* setActivation(const TRANS_FUN &value) {
    activation = value;
    return this;
  }

  LAYER_TYPE getType() const {
    return type;
  }

  MLPLayer* setType(const LAYER_TYPE &value) {
    type = value;
    return this;
  }


  int getNum_neurons() const {
    return num_neurons;
  }

  MLPLayer* setNum_neurons(int value) {
    num_neurons = value;
    return this;
  }

  int getNum_outputs() const {
    return num_outputs;
  }

  MLPLayer* setNum_outputs(int value) {
    num_outputs = value;
    return this;
  }

  Eigen::MatrixXd *getZ() const {
    return Z;
  }

  MLPLayer* setZ(Eigen::MatrixXd *value) {
    if(Z)
      Z->resize(0,0);
    Z = value;
    return this;
  }
  Eigen::MatrixXd *getS() const {
    return S;
  }

  MLPLayer* setS(Eigen::MatrixXd *value) {
    if(S){
        S->resize(0,0);
      }
    S = value;
    return this;
  }
  Eigen::MatrixXd *getD() const {
    return D;
  }

  MLPLayer* setD(Eigen::MatrixXd *value) {
    if(D)
      D->resize(0,0);
    D = value;
    return this;
  }


  Eigen::MatrixXd *getF() const {
    return F;
  }

  MLPLayer* setF(Eigen::MatrixXd *value) {
    if(F)
      F->resize(0,0);
    //      delete F;
    F = value;
    return this;
  }
  Eigen::MatrixXd *getW() const {
    return W;
  }

  MLPLayer* setW(Eigen::MatrixXd *value) {
    if(W)
      W->resize(0,0);
    W = value;
    return this;
  }
  int getBatch_size() const {
    return batch_size;
  }

  MLPLayer* setBatch_size(int value) {
    batch_size = value;
    return this;
  }


  void init() {
    Z = new Eigen::MatrixXd(batch_size, num_neurons);
    S = new Eigen::MatrixXd(batch_size, num_neurons);
    W = new Eigen::MatrixXd(num_neurons, num_outputs);
    D = new Eigen::MatrixXd(batch_size, num_neurons);
    F = new Eigen::MatrixXd(num_neurons,batch_size);


    *Z = Eigen::MatrixXd::Zero(batch_size, num_neurons);
    if(type != INPUT) {
        *S = Eigen::MatrixXd::Zero(batch_size, num_neurons);
        *D = Eigen::MatrixXd::Zero(batch_size, num_neurons);
      }
    if(type != OUTPUT) {
        // MatrixXd::Random returns uniform random numbers in (-1, 1).
        // Multiply by 2 to get the range of (-2, 2)
        *W = Eigen::MatrixXd::Random(num_neurons, num_outputs) * 2;
      }

    if(type == HIDDEN) {
        *F = Eigen::MatrixXd::Zero(num_neurons,batch_size);
      }
  }
private:
  static double logistic(double x) {
    return 1.0 / (1.0 + exp(-x));
  }

  static double logistic_deriv(double x) {
    return logistic(x) * (1.0 - logistic(x));
  }

  static inline double tanh_(double x) {
    return tanh(x);
  }

  static double tanh_deriv(double x) {
    double _tanh = tanh(x);
    return 1.0 - (_tanh * _tanh);
  }


};



