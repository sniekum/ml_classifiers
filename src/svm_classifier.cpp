/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Scott Niekum
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *********************************************************************/

/**
  * \author Scott Niekum
  */


#include "ml_classifiers/svm_classifier.h"
#include <pluginlib/class_list_macros.h>
#include <cmath>
#include <string>

PLUGINLIB_EXPORT_CLASS(ml_classifiers::SVMClassifier, ml_classifiers::Classifier)

namespace ml_classifiers{

    SVMClassifier::SVMClassifier(){}
    
    SVMClassifier::~SVMClassifier(){}
    
    void SVMClassifier::save(const std::string filename){}
    
    bool SVMClassifier::load(const std::string filename){return false;}
    
    void SVMClassifier::addTrainingPoint(std::string target_class, const std::vector<double> point)
    {
        class_data[target_class].push_back(point);
    }
    
    void SVMClassifier::train()
    {
        if(class_data.size() == 0){
            printf("SVMClassifier::train() -- No training data available! Doing nothing.\n");
            return;
        }
        
        int n_classes = class_data.size();
        
        //Count the training data
        int n_data = 0;
        int dims = class_data.begin()->second[0].size();
        for(ClassMap::iterator iter = class_data.begin(); iter != class_data.end(); iter++){
            CPointList cpl = iter->second;
            if(cpl.size() == 1)
                n_data += 2;    //There's a bug in libSVM for classes with only 1 data point, so we will duplicate them later
            else
                n_data += cpl.size();
        }
        
        //Allocate space for data in an svm_problem structure
        svm_data.l = n_data;
        svm_data.y = new double[n_data];
        svm_data.x = new svm_node*[n_data]; 
        for(int i=0; i<n_data; i++)
            svm_data.x[i] = new svm_node[dims+1];
        
        //Create maps between string labels and int labels
        label_str_to_int.clear();
        label_int_to_str.clear();
        int label_n = 0;
        for(ClassMap::iterator iter = class_data.begin(); iter != class_data.end(); iter++){
            std::string cname = iter->first;
            label_str_to_int[cname] = label_n;
            label_int_to_str[label_n] = cname;
            //cout << "MAP: " << label_n << "   " << cname << "   Size: " << iter->second.size() << endl;
            ++label_n;
        }
                
        //Find the range of the data in each dim and calc the scaling factors to scale from 0 to 1
        scaling_factors = new double*[dims];
        for(int i=0; i<dims; i++)
            scaling_factors[i] = new double[2];
            
        //Scale each dimension separately
        for(int j=0; j<dims; j++){
            //First find the min, max, and scaling factor
            double minval = INFINITY;
            double maxval = -INFINITY;
            for(ClassMap::iterator iter = class_data.begin(); iter != class_data.end(); iter++){
                CPointList cpl = iter->second;
                for(size_t i=0; i<cpl.size(); i++){
                    if(cpl[i][j] < minval) 
                        minval = cpl[i][j];
                    if(cpl[i][j] > maxval) 
                        maxval = cpl[i][j];
                }
            }
            double factor = maxval-minval;
            double offset = minval;
            
            //Do the scaling and save the scaling factor and offset
            for(ClassMap::iterator iter = class_data.begin(); iter != class_data.end(); iter++){
                for(size_t i=0; i<iter->second.size(); i++){
                    iter->second[i][j] = (iter->second[i][j] - offset) / factor;
                }
            }
            scaling_factors[j][0] = offset;
            scaling_factors[j][1] = factor;
        }
        
        //Put the training data into the svm_problem
        int n = 0;
        for(ClassMap::iterator iter = class_data.begin(); iter != class_data.end(); iter++){
            std::string cname = iter->first;
            CPointList cpl = iter->second;
            
            //Account for bug in libSVM with classes with only 1 data point by duplicating it.
            if(cpl.size() == 1){
                svm_data.y[n] = label_str_to_int[cname];
                svm_data.y[n+1] = label_str_to_int[cname];
                for(int j=0; j<dims; j++){
                    svm_data.x[n][j].index = j;
                    svm_data.x[n][j].value = cpl[0][j] + 0.001;
                    svm_data.x[n+1][j].index = j;
                    svm_data.x[n+1][j].value = cpl[0][j] + 0.001;
                }
                svm_data.x[n][dims].index = -1;
                svm_data.x[n+1][dims].index = -1;
                n = n + 2;
            }
            else{
                for(size_t i=0; i<cpl.size(); i++){
                    svm_data.y[n] = label_str_to_int[cname];
                    for(int j=0; j<dims; j++){
                        svm_data.x[n][j].index = j;
                        svm_data.x[n][j].value = cpl[i][j];
                    }
                svm_data.x[n][dims].index = -1;
                n = n + 1;
                }
            }
        } 
        
        //Set the training params
        svm_parameter params;
        params.svm_type = C_SVC;
        params.kernel_type = RBF;
        params.cache_size = 100.0;  
        params.gamma = 1.0;
        params.C = 1.0;
        params.eps = 0.001;
        params.shrinking = 1;
        params.probability = 0;
        params.degree = 0;
        params.nr_weight = 0;
        //params.weight_label = 
        //params.weight = 
        
        const char *err_str = svm_check_parameter(&svm_data, &params);
        if(err_str){
            printf("SVMClassifier::train() -- Bad SVM parameters!\n");
            printf("%s\n",err_str);
            return;
        }
        
        //Grid Search for best C and gamma params
        int n_folds = std::min(10, n_data);  //Make sure there at least as many points as folds
        double *resp = new double[n_data];
        double best_accy = 0.0;
        double best_g = 0.0;
        double best_c = 0.0;
        
        //First, do a coarse search
        for(double c = -5.0; c <= 15.0; c += 2.0){
            for(double g = 3.0; g >= -15.0; g -= 2.0){    
                params.gamma = pow(2,g);
                params.C = pow(2,c);
                
                svm_cross_validation(&svm_data, &params, n_folds, resp);
                
                //Figure out the accuracy using these params
                int correct = 0;
                for(int i=0; i<n_data; i++){
                    if(resp[i] == svm_data.y[i])
                        ++correct;
                    double accy = double(correct) / double(n_data);
                    if(accy > best_accy){
                        best_accy = accy;
                        best_g = params.gamma;
                        best_c = params.C;
                    }
                }
            }
        }
        
        //Now do a finer grid search based on coarse results   
        double start_c = best_c - 1.0;
        double end_c = best_c + 1.0;
        double start_g = best_g + 1.0;
        double end_g = best_g - 1.0;
        for(double c = start_c; c <= end_c; c += 0.1){
            for(double g = start_g; g >= end_g; g -= 0.1){
                params.gamma = pow(2,g);
                params.C = pow(2,c);
                svm_cross_validation(&svm_data, &params, n_folds, resp);
                
                //Figure out the accuracy using these params
                int correct = 0;
                for(int i=0; i<n_data; i++){
                    if(resp[i] == svm_data.y[i])
                        ++correct;
                    double accy = double(correct) / double(n_data);
                    
                    if(accy > best_accy){
                        best_accy = accy;
                        best_g = params.gamma;
                        best_c = params.C;
                    }
                }
            }
        }

        // Set params to best found in grid search
        params.gamma = best_g;
        params.C = best_c;
    
        printf("BEST PARAMS  ncl: %i   c: %f   g: %f   accy: %f \n\n", n_classes, best_c, best_g, best_accy);
        
        //Train the SVM
        trained_model = svm_train(&svm_data, &params);
    }
    
    void SVMClassifier::clear()
    {
        class_data.clear();
        label_str_to_int.clear();
        label_int_to_str.clear();
        trained_model = NULL;  // Actually, this should get freed
        scaling_factors = NULL;
    }
    
    std::string SVMClassifier::classifyPoint(const std::vector<double> point)
    {
        //Copy the point to be classified into an svm_node
        int dims = point.size();
        svm_node* test_pt = new svm_node[dims+1];
        for(int i=0; i<dims; i++){
            test_pt[i].index = i;
            //Scale the point using the training scaling values
            test_pt[i].value = (point[i]-scaling_factors[i][0]) / scaling_factors[i][1];
        }
        test_pt[dims].index = -1;
        
        //Classify the point using the currently trained SVM
        int label_n = svm_predict(trained_model, test_pt);
        return label_int_to_str[label_n];
    }
}

