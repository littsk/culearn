file(GLOB sub_dirs ${CMAKE_CURRENT_SOURCE_DIR}/*)

foreach(sub_dir ${sub_dirs})
    if(IS_DIRECTORY ${sub_dir})
        add_subdirectory(${sub_dir})
    endif()
endforeach()
