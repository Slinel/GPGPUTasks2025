#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libimages/debug_io.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "io/camera_reader.h"
#include "io/scene_reader.h"

#include "cpu_helpers/build_bvh_cpu.h"

#include "debug/debug_bvh.h"
#include "libimages/CImg.h"

#include <filesystem>
#include <fstream>

// Считает сколько непустых пикселей
template<typename T>
size_t countNonEmpty(const TypedImage<T> &image, T empty_value) {
    rassert(image.channels() == 1, 4523445132412, image.channels());
    size_t count = 0;
    #pragma omp parallel for reduction(+:count)
    for (ptrdiff_t j = 0; j < image.height(); ++j) {
        for (ptrdiff_t i = 0; i < image.width(); ++i) {
            if (image.ptr(j)[i] != empty_value) {
                ++count;
            }
        }
    }
    return count;
}

// Считает сколько отличающихся пикселей (отличающихся > threshold)
template<typename T>
size_t countDiffs(const TypedImage<T> &a, const TypedImage<T> &b, T threshold) {
    rassert(a.channels() == 1, 5634532413241, a.channels());
    rassert(a.channels() == b.channels(), 562435231453243);
    rassert(a.width() == b.width() && a.height() == b.height(), 562435231453243);
    size_t count = 0;
    #pragma omp parallel for reduction(+:count)
    for (ptrdiff_t j = 0; j < a.height(); ++j) {
        for (ptrdiff_t i = 0; i < a.width(); ++i) {
            if (std::abs(a.ptr(j)[i] - b.ptr(j)[i]) > threshold) {
                ++count;
            }
        }
    }
    return count;
}

void run(int argc, char** argv)
{
    // chooseGPUVkDevices:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    // TODO 000 сделайте здесь свой выбор API - если он отличается от OpenCL то в этой строке нужно заменить TypeOpenCL на TypeCUDA или TypeVulkan
    // TODO 000 после этого изучите этот код, запустите его, изучите соответсвующий вашему выбору кернел - src/kernels/<ваш выбор>/aplusb.<ваш выбор>
    // TODO 000 P.S. если вы выбрали CUDA - не забудьте установить CUDA SDK и добавить -DCUDA_SUPPORT=ON в CMake options
    // TODO 010 P.S. так же в случае CUDA - добавьте в CMake options (НЕ меняйте сами CMakeLists.txt чтобы не менять окружение тестирования):
    // TODO 010 "-DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo" (первое - чтобы включить поддержку WMMA, второе - чтобы compute-sanitizer и профилировщик знали номера строк кернела)
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    ocl::KernelSource ocl_rt_brute_force(ocl::getRTBruteForce());
    ocl::KernelSource ocl_rt_with_lbvh(ocl::getRTWithLBVH());

    avk2::KernelSource vk_rt_brute_force(avk2::getRTBruteForce());
    avk2::KernelSource vk_rt_with_lbvh(avk2::getRTWithLBVH());

    const std::string gnome_scene_path = "data/gnome/gnome.ply";
    std::vector<std::string> scenes = {
        gnome_scene_path,
        "data/powerplant/powerplant.obj",
        "data/san-miguel/san-miguel.obj",
    };

    const int niters = 10; // при отладке удобно запускать одну итерацию
    std::vector<double> gpu_rt_perf_mrays_per_sec;
    std::vector<double> gpu_lbvh_perfs_mtris_per_sec;

    std::cout << "Using " << AO_SAMPLES << " ray samples for ambient occlusion" << std::endl;
    for (std::string scene_path: scenes) {
        std::cout << "____________________________________________________________________________________________" << std::endl;
        timer total_t;
        if (scene_path == gnome_scene_path) {
            // data/gnome/gnome.ply содержится в репозитории, если он не нашелся - вероятно папка запуска настроена не верно
            rassert(std::filesystem::exists(scene_path), 3164718263781, "Probably wrong working directory?");
        } else if (!std::filesystem::exists(scene_path)) {
            std::cout << "Scene " << scene_path << " not found! Please download and unzip it for local evaluation - see link.txt" << std::endl;
            continue;
        }

        std::cout << "Loading scene " << scene_path << "..." << std::endl;
        timer loading_scene_t;
        SceneGeometry scene = loadScene(scene_path);
        // если на каком-то датасете падает - удобно взять подможество треугольников - например просто вызовите scene.faces.resize(10000);
        const unsigned int nvertices = scene.vertices.size();
        const unsigned int nfaces = scene.faces.size();
        rassert(nvertices > 0, 546345423523143);
        rassert(nfaces > 0, 54362452342);
        std::string scene_name = std::filesystem::path(scene_path).parent_path().filename().string();
        std::string camera_path = "data/" + scene_name + "/camera.txt";
        std::string results_dir = "results/" + scene_name;
        std::filesystem::create_directory(std::filesystem::path("results"));
        std::filesystem::create_directory(std::filesystem::path(results_dir));
        std::cout << "Loading camera " << camera_path << "..." << std::endl;
        CameraViewGPU camera = loadViewState(camera_path);
        const unsigned int width = camera.K.width;
        const unsigned int height = camera.K.height;
        double loading_data_time = loading_scene_t.elapsed();
        double images_saving_time = 0.0;
        std::cout << "Scene " << scene_name << " loaded: " << nvertices << " vertices, " << nfaces << " faces in " << loading_data_time << " sec" << std::endl;
        std::cout << "Camera framebuffer size: " << width << "x" << height << std::endl;

        // Аллоцируем буферы в VRAM
        gpu::gpu_mem_32f vertices_gpu(3 * nvertices);
        gpu::gpu_mem_32u faces_gpu(3 * nfaces);
        gpu::shared_device_buffer_typed<CameraViewGPU> camera_gpu(1);

        // Аллоцируем фрейм-буферы (то есть картинки в которые сохранится результат рендеринга)
        gpu::gpu_mem_32i framebuffer_face_id_gpu(width * height);
        gpu::gpu_mem_32f framebuffer_ambient_occlusion_gpu(width * height);

        // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
        timer pcie_writing_t;
        vertices_gpu.writeN((const float*) scene.vertices.data(), 3 * nvertices);
        faces_gpu.writeN((const unsigned int*) scene.faces.data(), 3 * nfaces);
        camera_gpu.writeN(&camera, 1);
        double pcie_writing_time = pcie_writing_t.elapsed();
        double pcie_reading_time = 0.0;

        // Перед каждой отрисовкой мы будем зачищать результирующие framebuffers этими значениями
        const int NO_FACE_ID = -1;
        const float NO_AMBIENT_OCCLUSION = -1.0f;
        double cleaning_framebuffers_time = 0.0;

        double brute_force_total_time = 0.0;
        image32i brute_force_framebuffer_face_ids;
        image32f brute_force_framebuffer_ambient_occlusion;
        const bool has_brute_force = (nfaces < 1000);
        if (has_brute_force) {
            std::vector<double> brute_force_times;
            for (int iter = 0; iter < niters; ++iter) {
                timer t;

                if (context.type() == gpu::Context::TypeOpenCL) {
                    ocl_rt_brute_force.exec(
                        gpu::WorkSize(16, 16, width, height),
                        vertices_gpu, faces_gpu,
                        framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu,
                        camera_gpu.clmem(), nfaces);
                } else {
                    rassert(false, 654724541234123);
                }

                brute_force_times.push_back(t.elapsed());
            }
            brute_force_total_time = stats::sum(brute_force_times);
            std::cout << "GPU brute force ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(brute_force_times) << std::endl;

            // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
            timer pcie_reading_t;
            brute_force_framebuffer_face_ids = image32i(width, height, 1);
            brute_force_framebuffer_ambient_occlusion = image32f(width, height, 1);
            framebuffer_face_id_gpu.readN(brute_force_framebuffer_face_ids.ptr(), width * height);
            framebuffer_ambient_occlusion_gpu.readN(brute_force_framebuffer_ambient_occlusion.ptr(), width * height);
            pcie_reading_time += pcie_reading_t.elapsed();

            size_t non_empty_brute_force_face_ids = countNonEmpty(brute_force_framebuffer_face_ids, NO_FACE_ID);
            size_t non_empty_brute_force_ambient_occlusion = countNonEmpty(brute_force_framebuffer_ambient_occlusion, NO_AMBIENT_OCCLUSION);
            rassert(non_empty_brute_force_face_ids > width * height / 10, 2345123412, non_empty_brute_force_face_ids);
            rassert(non_empty_brute_force_ambient_occlusion > width * height / 10, 3423413421, non_empty_brute_force_face_ids);
            timer images_saving_t;
            debug_io::dumpImage(results_dir + "/framebuffer_face_ids_brute_force.bmp", debug_io::randomMapping(brute_force_framebuffer_face_ids, NO_FACE_ID));
            debug_io::dumpImage(results_dir + "/framebuffer_ambient_occlusion_brute_force.bmp", debug_io::depthMapping(brute_force_framebuffer_ambient_occlusion));
            images_saving_time += images_saving_t.elapsed();
        }

        bool debug = false;
        double cpu_lbvh_time = 0.0;
        double rt_times_with_cpu_lbvh_sum = 0.0;
        int count = 0;
        point3f cMin;
        point3f cMax;
        std::vector<uint32_t> sorted_codes_cpu(nfaces);
        std::vector<uint32_t> leaf_faces_indices_cpu;
        std::vector<MortonCode> morton_codes_cpu(nfaces);
        std::vector<BVHNodeGPU> lbvh_nodes_cpu;

        {

            timer cpu_lbvh_t;
            unsigned int BVH_depth = buildLBVH_CPU(scene.vertices, scene.faces,
                lbvh_nodes_cpu,
                leaf_faces_indices_cpu,
                morton_codes_cpu,
                cMin,
                cMax,
                sorted_codes_cpu);

            if (debug) {
                //debug::dump_bvh_all_boxes_ply("results/bvh_all_cpu.ply", lbvh_nodes_cpu);
                debug::dump_bvh_boxes_ply("results/bvh_leaves_cpu.ply", lbvh_nodes_cpu, BVH_depth-2, BVH_depth);
            }

            cpu_lbvh_time = cpu_lbvh_t.elapsed();
            double build_mtris_per_sec = nfaces * 1e-6f / cpu_lbvh_time;
            std::cout << "CPU build LBVH in " << cpu_lbvh_time << " sec" << std::endl;
            std::cout << "CPU LBVH build performance: " << build_mtris_per_sec << " MTris/s" << std::endl;
            std::cout << "CPU BVH depth: " << BVH_depth << std::endl; //Хотел прикинуть какая глубина стека нужна
            if (debug) {
                std::cout<<"scene bounds computed by cpu: \n";
                std::cout << cMin.x <<' '<< cMin.y <<' '<< cMin.z << std::endl;
                std::cout << cMax.x <<' '<< cMax.y <<' '<< cMax.z << std::endl;
            }

            gpu::shared_device_buffer_typed<BVHNodeGPU> lbvh_nodes_gpu(lbvh_nodes_cpu.size());
            gpu::gpu_mem_32u leaf_faces_indices_gpu(leaf_faces_indices_cpu.size());
            lbvh_nodes_gpu.writeN(lbvh_nodes_cpu.data(), lbvh_nodes_cpu.size());
            leaf_faces_indices_gpu.writeN(leaf_faces_indices_cpu.data(), leaf_faces_indices_cpu.size());

            timer cleaning_framebuffers_t;
            framebuffer_face_id_gpu.fill(NO_FACE_ID);
            framebuffer_ambient_occlusion_gpu.fill(NO_AMBIENT_OCCLUSION);
            cleaning_framebuffers_time += cleaning_framebuffers_t.elapsed();

            std::vector<double> rt_times_with_cpu_lbvh;
            for (int iter = 0; iter < niters; ++iter) {
                timer t;

                if (context.type() == gpu::Context::TypeOpenCL) {
                    ocl_rt_with_lbvh.exec(
                        gpu::WorkSize(16, 16, width, height),
                        vertices_gpu, faces_gpu,
                        lbvh_nodes_gpu.clmem(), leaf_faces_indices_gpu.clmem(),
                        framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu,
                        camera_gpu.clmem(), nfaces);
                } else {
                    rassert(false, 654724541234123);
                }

                rt_times_with_cpu_lbvh.push_back(t.elapsed());
            }
            rt_times_with_cpu_lbvh_sum = stats::sum(rt_times_with_cpu_lbvh);
            double mrays_per_sec = width * height * AO_SAMPLES * 1e-6f / stats::median(rt_times_with_cpu_lbvh);
            std::cout << "GPU with CPU LBVH ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(rt_times_with_cpu_lbvh) << std::endl;
            std::cout << "GPU with CPU LBVH ray tracing performance: " << mrays_per_sec << " MRays/s" << std::endl;
            gpu_rt_perf_mrays_per_sec.push_back(mrays_per_sec);

            timer pcie_reading_t;
            image32i cpu_lbvh_framebuffer_face_ids(width, height, 1);
            image32f cpu_lbvh_framebuffer_ambient_occlusion(width, height, 1);
            framebuffer_face_id_gpu.readN(cpu_lbvh_framebuffer_face_ids.ptr(), width * height);
            framebuffer_ambient_occlusion_gpu.readN(cpu_lbvh_framebuffer_ambient_occlusion.ptr(), width * height);
            pcie_reading_time += pcie_reading_t.elapsed();

            timer cpu_lbvh_images_saving_t;
            debug_io::dumpImage(results_dir + "/framebuffer_face_ids_with_cpu_lbvh.bmp", debug_io::randomMapping(cpu_lbvh_framebuffer_face_ids, NO_FACE_ID));
            debug_io::dumpImage(results_dir + "/framebuffer_ambient_occlusion_with_cpu_lbvh.bmp", debug_io::depthMapping(cpu_lbvh_framebuffer_ambient_occlusion));
            images_saving_time += cpu_lbvh_images_saving_t.elapsed();
            if (has_brute_force) {
                unsigned int count_ao_errors = countDiffs(brute_force_framebuffer_ambient_occlusion, cpu_lbvh_framebuffer_ambient_occlusion, 0.01f);
                unsigned int count_face_id_errors = countDiffs(brute_force_framebuffer_face_ids, cpu_lbvh_framebuffer_face_ids, 1);
                rassert(count_ao_errors < width * height / 100, 345341512354123, count_ao_errors, to_percent(count_ao_errors, width * height));
                rassert(count_face_id_errors < width * height / 100, 3453415123546587, count_face_id_errors, to_percent(count_face_id_errors, width * height));
            }
        }

        double gpu_lbvh_time_sum = 0.0;
        double rt_times_with_gpu_lbvh_sum = 0.0;

        // TODO постройте LBVH на GPU
        // TODO оттрасируйте лучи на GPU используя построенный на GPU LBVH
        bool gpu_lbvg_gpu_rt_done = true;

        if (gpu_lbvg_gpu_rt_done) {

            gpu::shared_device_buffer_typed<MortonCode> morton_codes_gpu1(nfaces);
            gpu::shared_device_buffer_typed<MortonCode> morton_codes_gpu2(nfaces);
            gpu::gpu_mem_32u faces_indices_gpu1(nfaces);
            gpu::gpu_mem_32u faces_indices_gpu2(nfaces);
            gpu::gpu_mem_32u faces_parents(2*nfaces-1);
            gpu::gpu_mem_32f min_reduction1(nfaces*3);
            gpu::gpu_mem_32f max_reduction1(nfaces*3);
            gpu::gpu_mem_32f min_reduction2(nfaces*3);
            gpu::gpu_mem_32f max_reduction2(nfaces*3);
            gpu::shared_device_buffer_typed<BVHNodeGPU> lbvh_nodes(2*nfaces-1);
            gpu::shared_device_buffer_typed<MortonCode>* merge_code_from = &morton_codes_gpu1;
            gpu::gpu_mem_32u*                            merge_indx_from = &faces_indices_gpu1;
            gpu::shared_device_buffer_typed<MortonCode>* merge_code_to   = &morton_codes_gpu2;
            gpu::gpu_mem_32u*                            merge_indx_to   = &faces_indices_gpu2;

            ocl::KernelSource ocl_minmax_reduction(ocl::getMinMaxReduction());
            ocl::KernelSource ocl_reduction(ocl::getReduction());
            ocl::KernelSource ocl_build_morton(ocl::getBuildMorton());
            ocl::KernelSource ocl_merge_sort_morton(ocl::getMergeSortMorton());
            ocl::KernelSource ocl_setup_BVH_tree(ocl::getSetupBVHTree());
            ocl::KernelSource ocl_calculate_aabbs(ocl::getCalculateAABBs());

            std::vector<double> gpu_lbvh_times;
            for (int iter = 0; iter < 1; ++iter) {
            //for (int iter = 0; iter < niters; ++iter) {
                timer t;
                // 0 Этам
                // Найти границы сцены (сначала переиспользовал значения с цпу, но передумал, щас свертку напишу)
                ocl_minmax_reduction.exec(//это поиск центроидов, просто не переименовал
                            gpu::WorkSize(32, nfaces),
                            vertices_gpu,
                            faces_gpu,
                            min_reduction1,
                            max_reduction1,
                            nfaces);

                if (debug) {
                    std::vector<float> cMingpu2 = min_reduction1.readVector(nfaces*3);
                    float min_x = 100000000.0;
                    int counter = 0;
                    for (int i = 0; i+0<(nfaces*3); i+=3) {
                        if (cMingpu2[i+0] == -123.0) {counter+=1;}
                        min_x = fmin(min_x, cMingpu2[i+0]);
                    }std::cout<<"min_x_gpu: "<<min_x<<std::endl;
                    std::cout<<"counter: "<<counter<<std::endl;
                    float min_y = 100000000.0;
                    for (int i = 0; i+1<(nfaces*3); i+=3) {
                        min_y = fmin(min_y, cMingpu2[i+1]);
                    }std::cout<<"min_y_gpu: "<<min_y<<std::endl;
                    float min_z = 100000000.0;
                    for (int i = 0; i+2<(nfaces*3); i+=3) {
                        min_z = fmin(min_z, cMingpu2[i+2]);
                    }std::cout<<"min_z_gpu: "<<min_z<<std::endl;
                    std::vector<float> cMaxgpu2 = max_reduction1.readVector(nfaces);
                }


                gpu::gpu_mem_32f* min_from = &min_reduction1;
                gpu::gpu_mem_32f* min_to   = &min_reduction2;

                gpu::gpu_mem_32f* max_from = &max_reduction1;
                gpu::gpu_mem_32f* max_to   = &max_reduction2;
                unsigned int k = nfaces; //длинна в длинна в треугольниках
                bool flag = true;
                while (k>1) {

                    ocl_reduction.exec(
                            gpu::WorkSize(32, k),
                            *min_from,
                            *max_from,
                            *min_to,
                            *max_to,
                            k);

                    k = (k+31)/32;

                    if (flag) {
                        min_from = &min_reduction2;
                        min_to   = &min_reduction1;

                        max_from = &max_reduction2;
                        max_to   = &max_reduction1;
                    } else {
                        min_from = &min_reduction1;
                        min_to   = &min_reduction2;

                        max_from = &max_reduction1;
                        max_to   = &max_reduction2;
                    }
                    flag = !flag;
                }
                std::vector<float> cMingpu = min_from->readVector(3);
                std::vector<float> cMaxgpu = max_from->readVector(3);
                if (debug) {
                    std::cout<<"scene bounds computed by gpu: \n";
                    for (auto el : cMingpu) {std::cout<<el<<' ';}std::cout<<'\n';
                    for (auto el : cMaxgpu) {std::cout<<el<<' ';}std::cout<<'\n';
                }

                // 1 Этап
                // Для треугольников найти центроиды, построить для них коды мортона
                // Сохранить эти коды в массив кодов мортона соответствующих массиву faces_gpu
                ocl_build_morton.exec(
                        gpu::WorkSize(32, nfaces),
                        vertices_gpu,
                        faces_gpu,
                        morton_codes_gpu1,
                        faces_indices_gpu1,
                        nfaces,
                        cMingpu[0],cMingpu[1],cMingpu[2],
                        cMaxgpu[0],cMaxgpu[1],cMaxgpu[2]);


                if (debug) {
                    std::vector<MortonCode> morton_codes_gpu = morton_codes_gpu1.readVector(nfaces);
                    for (size_t q = 0; q < nfaces; q++) {
                        if (morton_codes_cpu[q]!=morton_codes_gpu[q]) {
                            std::cout<<"bs on "<<q<<"-th position (unsorted_codes)"<<'\n';
                            break;}}
                }

                // 2 Этап
                // Отсортировать коды, по этим же индексам отсортировать leaf_indices_gpu
                merge_code_from = &morton_codes_gpu1;
                merge_indx_from = &faces_indices_gpu1;
                merge_code_to   = &morton_codes_gpu2;
                merge_indx_to   = &faces_indices_gpu2;
                k = 1;
                while (k < nfaces) {
                    ocl_merge_sort_morton.exec(
                        gpu::WorkSize(GROUP_SIZE, nfaces),
                            *merge_code_from,
                            *merge_indx_from,
                            *merge_code_to,
                            *merge_indx_to,
                        k, nfaces);
                    k*=2;
                    if (merge_code_to == &morton_codes_gpu2) {
                        merge_code_from = &morton_codes_gpu2;
                        merge_indx_from = &faces_indices_gpu2;
                        merge_code_to = &morton_codes_gpu1;
                        merge_indx_to = &faces_indices_gpu1;
                    } else {
                        merge_code_from = &morton_codes_gpu1;
                        merge_indx_from = &faces_indices_gpu1;
                        merge_code_to = &morton_codes_gpu2;
                        merge_indx_to = &faces_indices_gpu2;
                    }
                }

                //Надо было с самого начала проверяться, все неправильно было
                if (debug) {
                    //работает ли сортировка лол, у меня идеи кончаются
                    std::vector<MortonCode> sorted_codes_gpu = merge_code_from->readVector(nfaces);
                    count = 0;
                    for (size_t q = 0; q < nfaces; q++) {
                        if (sorted_codes_cpu[q]!=sorted_codes_gpu[q]) {
                            std::cout<<"issue on idx: "<<q<<", cpu: "<<sorted_codes_cpu[q]
                            <<", gpu: "<<sorted_codes_gpu[q]<<" (sorted_codes) "<<'\n';
                            count++;
                            if (count>=10) {break;}}}

                    std::vector<uint32_t> sorted_face_indices_gpu = merge_indx_from->readVector(nfaces);
                    count = 0;
                    for (size_t q = 0; q < nfaces; q++) {
                        if (leaf_faces_indices_cpu[q]!=sorted_face_indices_gpu[q]) {
                            std::cout<<"issue on idx: "<<q<<", cpu: "<<leaf_faces_indices_cpu[q]
                            <<", gpu: "<<sorted_face_indices_gpu[q]<<" (sorted_indices) "<<'\n';
                            count++;
                            if (count>=10) {break;}}}
                }

                // 3 Этап
                // Постоить BVH (только структуру, без боксов)
                ocl_setup_BVH_tree.exec(
                    gpu::WorkSize(32, nfaces),
                        *merge_code_from,
                        lbvh_nodes.clmem(),
                        faces_parents,
                        nfaces);

                ocl_calculate_aabbs.exec(
                    gpu::WorkSize(32, nfaces),
                        lbvh_nodes.clmem(),
                        faces_parents,
                        *merge_indx_from,
                        vertices_gpu,
                        faces_gpu,
                        nfaces);

                if (debug) {
                    std::vector<BVHNodeGPU> bvh_nodes_gpu = lbvh_nodes.readVector(2*nfaces-1);
                    count = 0;
                    std::cout<<"faces count: "<<nfaces<<'\n';
                    for (size_t q = 0; q < (2*nfaces-1); q++) {
                        if (lbvh_nodes_cpu[q].leftChildIndex!=bvh_nodes_gpu[q].leftChildIndex ||
                            lbvh_nodes_cpu[q].rightChildIndex!=bvh_nodes_gpu[q].rightChildIndex) {
                            std::cout<<"issue on idx: "<<q<<", cpu: "<<lbvh_nodes_cpu[q].leftChildIndex
                                                          <<", gpu: "<< bvh_nodes_gpu[q].leftChildIndex    <<" (arranged_left_nodes) "  <<'\n';
                            std::cout<<"issue on idx: "<<q<<", cpu: "<<lbvh_nodes_cpu[q].rightChildIndex
                                                          <<", gpu: "<< bvh_nodes_gpu[q].rightChildIndex   <<" (arranged_right_nodes) " <<'\n';
                            count++;
                            if (count>=10) {break;}}

                        if (abs(lbvh_nodes_cpu[q].aabb.min_x - bvh_nodes_gpu[q].aabb.min_x)>0.001 ||
                            abs(lbvh_nodes_cpu[q].aabb.min_y - bvh_nodes_gpu[q].aabb.min_y)>0.001 ||
                            abs(lbvh_nodes_cpu[q].aabb.min_z - bvh_nodes_gpu[q].aabb.min_z)>0.001) {
                            std::cout<<"issue on idx: "<<q<<", cpu: "<<lbvh_nodes_cpu[q].aabb.min_y
                                                          <<", gpu: "<< bvh_nodes_gpu[q].aabb.min_y    <<" (box size) "  <<'\n';
                            count++;
                            //if (count>=100) {break;}
                            }}

                    //debug::dump_bvh_all_boxes_ply("results/bvh_all_пpu.ply", lbvh_nodes_cpu);
                    std::vector<BVHNodeGPU> nodes_gpu = lbvh_nodes.readVector(nfaces);
                    debug::dump_bvh_boxes_ply("results/bvh_leaves_gpu.ply", nodes_gpu, 18-2, 18);
                }

                gpu_lbvh_times.push_back(t.elapsed());

            }
            gpu_lbvh_time_sum = stats::sum(gpu_lbvh_times);
            double build_mtris_per_sec = nfaces * 1e-6f / stats::median(gpu_lbvh_times);
            std::cout << "GPU LBVH build times (in seconds) - " << stats::valuesStatsLine(gpu_lbvh_times) << std::endl;
            std::cout << "GPU LBVH build performance: " << build_mtris_per_sec << " MTris/s" << std::endl;
            gpu_lbvh_perfs_mtris_per_sec.push_back(build_mtris_per_sec);

            timer cleaning_framebuffers_t;
            framebuffer_face_id_gpu.fill(NO_FACE_ID);
            framebuffer_ambient_occlusion_gpu.fill(NO_AMBIENT_OCCLUSION);
            cleaning_framebuffers_time += cleaning_framebuffers_t.elapsed();

            std::vector<double> gpu_lbvh_rt_times;
            for (int iter = 0; iter < niters; ++iter) {
                timer t;

                ocl_rt_with_lbvh.exec(
                        gpu::WorkSize(16, 16, width, height),
                        vertices_gpu, faces_gpu,
                        lbvh_nodes.clmem(), *merge_indx_from,
                        framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu,
                        camera_gpu.clmem(), nfaces);

                gpu_lbvh_rt_times.push_back(t.elapsed());
            }
            rt_times_with_gpu_lbvh_sum = stats::sum(gpu_lbvh_rt_times);
            double mrays_per_sec = width * height * AO_SAMPLES * 1e-6f / stats::median(gpu_lbvh_rt_times);
            std::cout << "GPU with GPU LBVH ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(gpu_lbvh_rt_times) << std::endl;
            std::cout << "GPU with GPU LBVH ray tracing performance: " << mrays_per_sec << " MRays/s" << std::endl;
            gpu_rt_perf_mrays_per_sec.push_back(mrays_per_sec);

            timer pcie_reading_t;
            image32i gpu_lbvh_framebuffer_face_ids(width, height, 1);
            image32f gpu_lbvh_framebuffer_ambient_occlusion(width, height, 1);
            framebuffer_face_id_gpu.readN(gpu_lbvh_framebuffer_face_ids.ptr(), width * height);
            framebuffer_ambient_occlusion_gpu.readN(gpu_lbvh_framebuffer_ambient_occlusion.ptr(), width * height);
            pcie_reading_time += pcie_reading_t.elapsed();

            timer gpu_lbvh_images_saving_t;
            debug_io::dumpImage(results_dir + "/framebuffer_face_ids_with_gpu_lbvh.bmp", debug_io::randomMapping(gpu_lbvh_framebuffer_face_ids, NO_FACE_ID));
            debug_io::dumpImage(results_dir + "/framebuffer_ambient_occlusion_with_gpu_lbvh.bmp", debug_io::depthMapping(gpu_lbvh_framebuffer_ambient_occlusion));
            images_saving_time += gpu_lbvh_images_saving_t.elapsed();

            bool denoising = true;
            if (denoising) {
                gpu::gpu_mem_32f framebuffer_denoised(width * height);
                ocl::KernelSource ocl_denoise(ocl::getDenoiser());
                ocl_denoise.exec(
                            gpu::WorkSize(32, 1, width, height),
                            framebuffer_ambient_occlusion_gpu,
                            framebuffer_denoised,
                            width, height);

                image32f gpu_lbvh_framebuffer_ambient_occlusion_denoised(width, height, 1);
                framebuffer_denoised.readN(gpu_lbvh_framebuffer_ambient_occlusion_denoised.ptr(), width * height);
                debug_io::dumpImage(results_dir + "/framebuffer_ambient_occlusion_with_gpu_lbvh_denoised.bmp",
                            debug_io::depthMapping(gpu_lbvh_framebuffer_ambient_occlusion_denoised));
            }

            if (has_brute_force) {
                unsigned int count_ao_errors = countDiffs(brute_force_framebuffer_ambient_occlusion, gpu_lbvh_framebuffer_ambient_occlusion, 0.01f);
                unsigned int count_face_id_errors = countDiffs(brute_force_framebuffer_face_ids, gpu_lbvh_framebuffer_face_ids, 1);
                rassert(count_ao_errors < width * height / 100, 3567856512354123, count_ao_errors, to_percent(count_ao_errors, width * height));
                rassert(count_face_id_errors < width * height / 100, 3453465346387, count_face_id_errors, to_percent(count_face_id_errors, width * height));
            }
        }

        double total_time = total_t.elapsed();
        std::cout << "Scene processed in " << total_t.elapsed() << " sec = ";
        std::cout << to_percent(loading_data_time, total_time) << " scene IO + ";
        if (has_brute_force) {
            std::cout << to_percent(brute_force_total_time, total_time) << " brute force RT + ";
        }
        std::cout << to_percent(cpu_lbvh_time, total_time) << " CPU LBVH + ";
        std::cout << to_percent(rt_times_with_cpu_lbvh_sum, total_time) << " GPU with CPU LBVH + ";
        std::cout << to_percent(gpu_lbvh_time_sum, total_time) << " GPU LBVH + ";
        std::cout << to_percent(rt_times_with_gpu_lbvh_sum, total_time) << " GPU with GPU LBVH + ";
        std::cout << to_percent(images_saving_time, total_time) << " images IO + ";
        std::cout << to_percent(pcie_writing_time, total_time) << " PCI-E write + ";
        std::cout << to_percent(pcie_reading_time, total_time) << " PCI-E read + ";
        std::cout << to_percent(cleaning_framebuffers_time, total_time) << " cleaning VRAM";
        std::cout << std::endl;
    }

    std::cout << "____________________________________________________________________________________________" << std::endl;
    double avg_gpu_rt_perf = stats::avg(gpu_rt_perf_mrays_per_sec);
    double avg_lbvh_build_perf = stats::avg(gpu_lbvh_perfs_mtris_per_sec);
    std::cout << "Total GPU RT with  LBVH avg perf: " << avg_gpu_rt_perf << " MRays/sec (all " << stats::vectorToString(gpu_rt_perf_mrays_per_sec) << ")" << std::endl;
    std::cout << "Total building GPU LBVH avg perf: " << avg_lbvh_build_perf << " MTris/sec (all " << stats::vectorToString(gpu_lbvh_perfs_mtris_per_sec) << ")" << std::endl;
    std::cout << "Final score: " << avg_gpu_rt_perf * avg_lbvh_build_perf << " coolness" << std::endl;
    if (gpu_rt_perf_mrays_per_sec.size() != 6 || gpu_lbvh_perfs_mtris_per_sec.size() != 3) {
        std::cout << "Results are incomplete!" << std::endl;
    }
}

int main(int argc, char** argv)
{
    try {
        run(argc, argv);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (e.what() == DEVICE_NOT_SUPPORT_API) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за выбора CUDA API (его нет на процессоре - т.е. в случае CI на GitHub Actions)
            return 0;
        } if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за того что задание еще не выполнено
            return 0;
        } else {
            // Выставляем ненулевой exit code, чтобы сообщить, что случилась ошибка
            return 1;
        }
    }

    return 0;
}
