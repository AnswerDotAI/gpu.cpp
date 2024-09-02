# file name to find
set(FILENAME "gpu.hpp")

# Function to check for file existence up the directory hierarchy
function(find_project_root current_dir filename result_var)
    set(found FALSE) # Flag to indicate if the file is found
    set(current_check_dir "${current_dir}") # Start from the given directory
    # using 1 is jsut to supress the cmane-format warning
    foreach(i RANGE 0 2 1)
        set(filepath "${current_check_dir}/${filename}")

        if(EXISTS "${filepath}")
            set(${result_var}
                "${current_check_dir}"
                PARENT_SCOPE)
            set(found TRUE)
            break()
        endif()

        # Move one level up
        get_filename_component(current_check_dir "${current_check_dir}"
                               DIRECTORY)
    endforeach()

    if(NOT found)
        set(${result_var}
            ""
            PARENT_SCOPE) # Set to empty if not found
    endif()
endfunction()
