include(GoogleTest)

file(GLOB_RECURSE TEST_SOURCES *.cpp *.hpp)

add_executable(mlengine_test ${TEST_SOURCES})

target_link_libraries(mlengine_test
    PRIVATE
        GTest::gtest_main
        mlengine_core
)

gtest_discover_tests(mlengine_test)
