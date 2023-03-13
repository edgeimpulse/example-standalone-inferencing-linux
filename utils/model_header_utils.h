#ifndef MODEL_HEADER_UTILS_H_
#define MODEL_HEADER_UTILS_H_

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <errno.h>
#include <libgen.h> // required for dirname and basename

// https://gist.github.com/JonathonReinhart/8c0d90191c38af2dcadb102c4e202950
/* Make a directory; already existing dir okay */
static int maybe_mkdir(const char* path, mode_t mode)
{
    struct stat st;
    errno = 0;

    /* Try to make the directory */
    if (mkdir(path, mode) == 0)
        return 0;

    /* If it fails for any reason but EEXIST, fail */
    if (errno != EEXIST)
        return -1;

    /* Check if the existing path is a directory */
    if (stat(path, &st) != 0)
        return -1;

    /* If not, fail with ENOTDIR */
    if (!S_ISDIR(st.st_mode)) {
        errno = ENOTDIR;
        return -1;
    }

    errno = 0;
    return 0;
}

int mkdir_p(const char *path)
{
    /* Adapted from http://stackoverflow.com/a/2336245/119527 */
    char *_path = NULL;
    char *p;
    int result = -1;
    mode_t mode = 0777;

    errno = 0;

    /* Copy string so it's mutable */
    _path = strdup(path);
    if (_path == NULL)
        goto out;

    /* Iterate the string */
    for (p = _path + 1; *p; p++) {
        if (*p == '/') {
            /* Temporarily truncate */
            *p = '\0';

            if (maybe_mkdir(_path, mode) != 0)
                goto out;

            *p = '/';
        }
    }

    if (maybe_mkdir(_path, mode) != 0)
        goto out;

    result = 0;

out:
    free(_path);
    return result;
}

char* get_basename(const char *path) {
    char* basec = strdup(path);
    return basename(basec);
}

char* get_dirname(const char *path) {
    char* dirc = strdup(path);
    return dirname(dirc);
}

bool dir_exists(char *pathname)
{
    bool ret_value = false;
    struct stat info;

    if( stat( pathname, &info ) != 0 ) {
        ei_printf( "cannot access %s\n", pathname );
    } else if( info.st_mode & S_IFDIR ) {
        ei_printf( "%s is a directory\n", pathname );
        ret_value = true;
    } else {
        ei_printf( "%s is no directory\n", pathname );
    }

    return ret_value;
}

bool file_exists(char *model_file_name)
{
    FILE *file = fopen(model_file_name, "r");
    if (file) {
        fclose(file);
        return true;
    }
    else {
        return false;
    }
}

bool create_project_if_not_exists(std::string project_path, const ei_model_h_files* proj, unsigned int elems) {
    if (!dir_exists((char*)project_path.c_str())) {
        ei_printf("INFO: Model dir '%s' does not exist, creating now. \n", project_path.c_str());
        int rc = mkdir_p(project_path.c_str());
        (void) rc;
        //fprintf(stderr, "mkdir_p(\"%s\") returned %d: %m\n", project_path, rc);

        for (unsigned int f = 0; f < elems; f++) {

            // create new filepath
            std::string proj_filepath = project_path + "/" + std::string(proj[f].filename);

            char *dirname = get_dirname(proj_filepath.c_str());
            //printf("dirname: %s\n", dirname);

            int rc = mkdir_p(dirname);
            (void) rc;
            //fprintf(stderr, "mkdir_p(\"%s\") returned %d: %m\n", dirname, rc);

            FILE *file = fopen(proj_filepath.c_str(), "wb");
            fwrite(proj[f].buffer, proj[f].buf_len, 1, file);
            fclose(file);

            free(dirname);
        }
    }
    return true;
}


#endif // MODEL_HEADER_UTILS_H_
