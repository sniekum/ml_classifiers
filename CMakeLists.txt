# http://ros.org/doc/groovy/api/catkin/html/user_guide/supposed.html
cmake_minimum_required(VERSION 2.8.3)
project(ml_classifiers)

find_package(catkin REQUIRED
  COMPONENTS message_generation std_msgs roscpp pluginlib)
find_package(Eigen3 REQUIRED)

set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})

add_message_files(FILES
  ClassDataPoint.msg
  DIRECTORY msg)

add_service_files(FILES
  AddClassData.srv ClassifyData.srv 	ClearClassifier.srv 	CreateClassifier.srv
  LoadClassifier.srv SaveClassifier.srv TrainClassifier.srv
  DIRECTORY srv)

generate_messages(DEPENDENCIES std_msgs)

catkin_package(
  DEPENDS Eigen3
  CATKIN_DEPENDS message_runtime std_msgs roscpp pluginlib 
  INCLUDE_DIRS include
  LIBRARIES ZeroClassifier NearestNeighborClassifier SVMClassifier
)

add_definitions(${EIGEN3_DEFINITIONS})

include_directories(include ${EIGEN3_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})

add_library(ZeroClassifier src/zero_classifier.cpp) 
add_library(NearestNeighborClassifier src/nearest_neighbor_classifier.cpp)
add_library(SVMClassifier src/svm_classifier.cpp src/libsvm-3.14/svm.cpp)
add_executable(classifier_server nodes/classifier_server.cpp)
target_link_libraries(classifier_server ${catkin_LIBRARIES})
add_dependencies(classifier_server ${PROJECT_NAME}_generate_messages_cpp)

install(TARGETS classifier_server ZeroClassifier NearestNeighborClassifier SVMClassifier
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
