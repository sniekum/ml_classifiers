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

#include "ml_classifiers/zero_classifier.h"
#include "ml_classifiers/nearest_neighbor_classifier.h"
#include "ml_classifiers/CreateClassifier.h"
#include "ml_classifiers/AddClassData.h"
#include "ml_classifiers/TrainClassifier.h"
#include "ml_classifiers/ClearClassifier.h"
#include "ml_classifiers/SaveClassifier.h"
#include "ml_classifiers/LoadClassifier.h"
#include "ml_classifiers/ClassifyData.h"
#include <pluginlib/class_loader.h>

using namespace ml_classifiers;
using namespace std;


map<string, boost::shared_ptr<Classifier>> classifier_list;
pluginlib::ClassLoader<Classifier> c_loader("ml_classifiers", "ml_classifiers::Classifier");

bool createHelper(string class_type, boost::shared_ptr<Classifier>& c)
{
    try{
        c = c_loader.createInstance(class_type);
    }
    catch(pluginlib::PluginlibException& ex){
        ROS_ERROR("Classifer plugin failed to load! Error: %s", ex.what());
    }
    
    return true;
}


bool createCallback(CreateClassifier::Request &req,
                    CreateClassifier::Response &res )
{
    string id = req.identifier;
    boost::shared_ptr<Classifier> c;
    
    if (!createHelper(req.class_type, c)){
        res.success = false;
        return false;
    }
    
    if (classifier_list.find(id) != classifier_list.end()){
        cout << "WARNING: ID already exists, overwriting: " << req.identifier << endl;
        classifier_list.erase(id);
    }
    classifier_list[id] = c;
    
    res.success = true;
    return true;
}


bool addCallback(AddClassData::Request &req,
                 AddClassData::Response &res )
{
    string id = req.identifier;
    if(classifier_list.find(id) == classifier_list.end()){    
        res.success = false;
        return false;
    }
    
    for(size_t i=0; i<req.data.size(); i++)
        classifier_list[id]->addTrainingPoint(req.data[i].target_class, req.data[i].point);
    
    res.success = true;
    return true;
}


bool trainCallback(TrainClassifier::Request &req,
                   TrainClassifier::Response &res )
{
    string id = req.identifier;
    if(classifier_list.find(id) == classifier_list.end()){    
        res.success = false;
        return false;
    }
    
    cout << "Training " << id << endl;
    
    classifier_list[id]->train();
    res.success = true;
    return true;
}


bool clearCallback(ClearClassifier::Request &req,
                   ClearClassifier::Response &res )
{
    string id = req.identifier;
    if(classifier_list.find(id) == classifier_list.end()){    
        res.success = false;
        return false;
    }
    
    classifier_list[id]->clear();
    res.success = true;
    return true;
}


bool saveCallback(SaveClassifier::Request &req,
                  SaveClassifier::Response &res )
{
    string id = req.identifier;
    if(classifier_list.find(id) == classifier_list.end()){    
        res.success = false;
        return false;
    }
    
    classifier_list[id]->save(req.filename);
    res.success = true;
    return true;
}


bool loadCallback(LoadClassifier::Request &req,
                  LoadClassifier::Response &res )
{
    string id = req.identifier;
    
    boost::shared_ptr<Classifier> c;
    if(!createHelper(req.class_type, c)){
        res.success = false;
        return false;
    }
    
    if(!c->load(req.filename)){
        res.success = false;
        return false;
    }
    
    if(classifier_list.find(id) != classifier_list.end()){
        cout << "WARNING: ID already exists, overwriting: " << req.identifier << endl;
        classifier_list.erase(id);
    }
    classifier_list[id] = c;
    
    res.success = true;
    return true;
}


bool classifyCallback(ClassifyData::Request &req,
                      ClassifyData::Response &res )
{
    string id = req.identifier;
    for(size_t i=0; i<req.data.size(); i++){
        string class_num = classifier_list[id]->classifyPoint(req.data[i].point);
        res.classifications.push_back(class_num);
    }
    
    return true;
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "classifier_server");
    ros::NodeHandle n;
    
    ros::ServiceServer service1 = n.advertiseService("/ml_classifiers/create_classifier", createCallback);
    ros::ServiceServer service2 = n.advertiseService("/ml_classifiers/add_class_data", addCallback);
    ros::ServiceServer service3 = n.advertiseService("/ml_classifiers/train_classifier", trainCallback);
    ros::ServiceServer service4 = n.advertiseService("/ml_classifiers/clear_classifier", clearCallback);
    ros::ServiceServer service5 = n.advertiseService("/ml_classifiers/save_classifier", saveCallback);
    ros::ServiceServer service6 = n.advertiseService("/ml_classifiers/load_classifier", loadCallback);
    ros::ServiceServer service7 = n.advertiseService("/ml_classifiers/classify_data", classifyCallback);
    
    ROS_INFO("Classifier services now ready");
    ros::spin();

    return 0;
}
