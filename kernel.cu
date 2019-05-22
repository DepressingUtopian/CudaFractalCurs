
#include <iostream>

// GLEW
#define GLEW_STATIC
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include <stdio.h>

const GLuint WIDTH = 800, HEIGHT = 600;
const GLuint MAX_ITERATION = 500;
const GLdouble MAX_R = -10.0,MIN_R = 10.0,MAX_I = 10.0,MIN_I = -10.0;
/* OpenGL interoperability */
dim3 blocks, threads;

GLuint vbo; //int указатель на Vertex Buffer Object

GLuint VBO;
GLuint VAO;

struct cudaGraphicsResource *cuda_vbo_resource; //Структура предоставляющая реализацию VBO в CUDA

float* dev_screen; //Память представляющая цвет пикселей на экране
const char* vertexShaderSource =
"#version 330 core\n"
"layout (location = 0) in vec3 position;\n"
"layout (location = 1) in vec3 color;\n"
"out vec3 ourColor\;\n"
"void main () {\n"
"  gl_Position = vec4 (position, 1.0);\n"
"  ourColor = color;\n"
"}";

const char* fragmentShaderSource =
"#version 330 core"
"in vec3 ourColor;\n"
"out vec4 color;\n"
"void main () {\n"
"  color = vec4(ourColor, 1.0f);\n"
"}\n";

GLuint vertexShader;
GLuint fragmentShader;

GLuint shaderProgram;
//Отлов ошибок CUDA
static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	// Когда пользователь нажимает ESC, мы устанавливаем свойство WindowShouldClose в true, 
	// и приложение после этого закроется
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
}
__device__ double mapToReal(int x, int windowWidth, double minR, double maxR)
{
	double range = maxR - minR;
	return x * (range / windowWidth) + minR;
}
__device__ double mapToImaginary(int y, int windowHeigth, double minI, double maxI)
{
	double range = maxI - minI;
	return y * (range / windowHeigth) + minI;
}
__device__ int MandelbrotFunction(double p, double q, int maxIteration)
{
	int i = 0;
	double x_t = 0.0, y_t = 0.0;
	while (i < maxIteration && x_t*x_t - y_t * y_t < 4.0)
	{
		double temp = x_t * x_t - y_t * y_t + p;
		y_t = 2.0 * x_t * y_t + q;
		i++;
	}

	return i;
}
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,unsigned int vbo_res_flags) 
{
	unsigned int size = WIDTH * HEIGHT * sizeof(float) * 6;

	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);

	HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

//	glVertexAttribPointer(1, 4, sizeof(uchar4), GL_FALSE,size, (GLvoid*)0);
//	glEnableVertexAttribArray(0);
}

void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res) 
{
	HANDLE_ERROR(cudaGraphicsUnregisterResource(cuda_vbo_resource));

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}
__device__ void setCoord(int x,int y,int offset,float *pixels)
{
	pixels[offset * 6] = x;			//x
	pixels[offset * 6 + 1] = x;		//y
	pixels[offset * 6 + 2] = 0.0f;	//z

}
__device__ void setColor(int offset,int ColorValue, float *pixels)
{
	
	pixels[offset * 6 + 3] = (ColorValue % 256);			//x
	pixels[offset * 6 + 4] = (ColorValue % 256);			//y
	pixels[offset * 6 + 5] = (ColorValue % 256);			//z

}
__global__ void MandelbrotKernel(float* screen,int windowHeigth,int windowWidth,double maxR,double maxI,double minR,double minI,int maxIteration)
{
	int x = blockIdx.x;
	int y = blockIdx.y;

	if(x >= WIDTH || y >= HEIGHT)
		return;

	int offset = x + y * gridDim.x;
	double p = mapToReal(x, windowWidth,minR,maxR);
	double q = mapToImaginary(y, windowHeigth, minI, maxI);
	int MandelbrotValue = MandelbrotFunction(p, q, maxIteration);

	setCoord(x, y,offset,screen);
	setColor(offset,MandelbrotValue, screen);

}

void InitOpenGL()
{
	//Инициализация GLFW
	glfwInit();


	//Настройка GLFW
	//Задается минимальная требуемая версия OpenGL. 
	//Мажорная 
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	//Минорная
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	//Установка профайла для которого создается контекст
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//Выключение возможности изменения размера окна
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
}
void initCuda(int deviceId) {
	int deviceCount = 0;
	HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));

	if (deviceCount <= 0) {
		printf("No CUDA devices found\n");
		exit(-1);
	}

	HANDLE_ERROR(cudaGLSetGLDevice(deviceId));

	cudaDeviceProp properties;
	HANDLE_ERROR(cudaGetDeviceProperties(&properties, deviceId));

	threads.x = 32;
	threads.y = properties.maxThreadsPerBlock / threads.x - 2; // to avoid cudaErrorLaunchOutOfResources error

	blocks.x = (WIDTH + threads.x - 1) / threads.x;
	blocks.y = (HEIGHT + threads.y - 1) / threads.y;

	printf(
		"Debug: blocks(%d, %d), threads(%d, %d)\nCalculated Resolution: %d x %d\n",
		blocks.x, blocks.y, threads.x, threads.y, blocks.x * threads.x,
		blocks.y * threads.y);
}
void MaldelbrotGPU_Calculation()
{
	
	size_t size;

	HANDLE_ERROR(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	HANDLE_ERROR(
		cudaGraphicsResourceGetMappedPointer((void**)&dev_screen, &size, cuda_vbo_resource));
	cudaEvent_t startEvent, stopEvent;
	float elapsedTime = 0.0f;
	HANDLE_ERROR(cudaEventCreate(&startEvent));
	HANDLE_ERROR(cudaEventCreate(&stopEvent));
	HANDLE_ERROR(cudaEventRecord(startEvent, 0));

	// Render Image
	MandelbrotKernel << <blocks, threads >> > (dev_screen, HEIGHT, WIDTH, MAX_R, MAX_I, MIN_R, MIN_I, MAX_ITERATION);
	HANDLE_ERROR(cudaDeviceSynchronize());

	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

	// Kernel Time measure
	HANDLE_ERROR(cudaEventRecord(stopEvent, 0));
	HANDLE_ERROR(cudaEventSynchronize(stopEvent));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

	std::cout << std::endl;
	printf("Время выполнения: %f ms\n", elapsedTime);

}
void CreateVertexShader()
{
	GLint success;
	GLchar infoLog[512];
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);

	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
}
void CreateFragmentShader()
{
	GLint success;
	GLchar infoLog[512];
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);

	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
}
void InitShaders()
{
	GLint success;
	GLchar infoLog[512];

	CreateVertexShader();
	CreateFragmentShader();

	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}
int main()
{
	setlocale(LC_ALL,"Russian");
	InitOpenGL();
	//////////////////////////////////////
	
	
	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "MaldelbrotFration", nullptr, nullptr);
	if (window == nullptr)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	//Инициализация GLEW
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK)
	{
		std::cout << "Failed to initialize GLEW" << std::endl;
		return -1;
	}

	//Установка буфера отрисовки
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);

	glViewport(0, 0, width, height);

	//Регистрации функции обратного вызова
	glfwSetKeyCallback(window, key_callback);

	initCuda(0);
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	InitShaders();

	MaldelbrotGPU_Calculation();
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);
	//Окно существует пока не будет закрыто


	/*

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);

	// 0. Копируем массив с вершинами в буфер OpenGL
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	// 1. Затем установим указатели на вершинные атрибуты
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
	// 2. Используем нашу шейдерную программу
	glUseProgram(shaderProgram);
	*/
	/*glGenVertexArrays(1, &VAO);
	// ..:: Код инициализации (выполняется единожды (если, конечно, объект не будет часто изменяться)) :: .. 
// 1. Привязываем VAO
	glBindVertexArray(VAO);
	// 2. Копируем наш массив вершин в буфер для OpenGL
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(dev_screen), dev_screen, GL_DYNAMIC_DRAW);
	// 3. Устанавливаем указатели на вершинные атрибуты 
	
	//4. Отвязываем VAO
	glBindVertexArray(0);*/

	
	// ..:: Код отрисовки (в игровом цикле) :: ..
	// 5. Отрисовываем объект




	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
	
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		
		glUseProgram(shaderProgram);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glDrawArrays(GL_POINT, 0, 3);
		glBindVertexArray(0);

		glfwSwapBuffers(window); /* Updates the screen */
		glfwWaitEvents(); /* Polls input */
	}

	
	//Очистка мусора
	glfwTerminate();
	return 0;
}
