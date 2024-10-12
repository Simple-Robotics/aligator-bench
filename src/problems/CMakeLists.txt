add_library(test_problems SHARED linear-problem.cpp ur5-util.cpp)
target_include_directories(test_problems PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_problems PUBLIC ${PROJECT_NAME})
