// CPU_Mandelbrot.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include "pch.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <limits>
#include <thread>
#include <future>
#include <fstream>
// GLEW
#define GLEW_STATIC
#include <GL/glew.h>
#include <time.h> 

// GLFW
#include <GLFW/glfw3.h>


#include <stdio.h>


#include <sys/stat.h>

using namespace std;

bool asyncHandler = true;
// Function prototypes
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);

const GLuint WIDTH = 1920, HEIGHT = 1080;
GLfloat cpu_PixelsMemory[WIDTH * HEIGHT * sizeof(GLfloat) * 6];
GLdouble screen_ratio = (double)WIDTH / (double)HEIGHT;

double cx = 0.0, cy = 0.0, zoom = 1.0;
int fps = 0;
bool isChange = false;
GLFWwindow *window = nullptr;
GLuint shaderProgram;
float elapsed_time = 0;

template<typename T>
bool future_is_ready(std::future<T>& t){
    return t.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

const GLchar* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 position;\n"
"layout (location = 1) in vec3 color;\n"
"out vec3 ourColor;\n"
"void main()\n"
"{\n"
"gl_Position = vec4(position, 1.0);\n"
"ourColor = color;\n"
"}\0";
const GLchar* fragmentShaderSource = "#version 330 core\n"
"in vec3 ourColor;\n"
"out vec4 color;\n"
"void main()\n"
"{\n"
"color = vec4(ourColor, 1.0f);\n"
"}\n\0";

GLuint MAX_ITERATION = 1;

//GLuint vbo; //int указатель на Vertex Buffer Object


struct cudaGraphicsResource *cuda_vbo_resource; //Структура предоставляющая реализацию VBO в CUDA


double last_time = 0, current_time = 0;
unsigned int ticks = 0;

bool keys[1024] = { 0 };


static void cursor_callback(GLFWwindow* window, double xpos, double ypos)
{
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	double xr = 2.0 * (xpos / (double)WIDTH - 0.5);
	double yr = 2.0 * (ypos / (double)HEIGHT - 0.5);

	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		cx += (xr - cx) / zoom / 2.0;
		cy -= (yr - cy) / zoom / 2.0;
		isChange = true;
	}
	//isChange = true;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	zoom += yoffset * 0.1 * zoom;
	if (zoom < 0.1) {
		zoom = 0.1;

	}
	isChange = true;
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	const double d = 0.1 / zoom;

	if (action == GLFW_PRESS) {
		keys[key] = true;
		isChange = true;
	}
	else if (action == GLFW_RELEASE) {
		keys[key] = false;
		isChange = true;
	}

	if (keys[GLFW_KEY_ESCAPE]) {
		glfwSetWindowShouldClose(window, 1);

	}
	else if (keys[GLFW_KEY_A]) {
		cx -= d;
		isChange = true;
	}
	else if (keys[GLFW_KEY_D]) {
		cx += d;
		isChange = true;
	}
	else if (keys[GLFW_KEY_W]) {
		cy += d;
		isChange = true;
	}
	else if (keys[GLFW_KEY_S]) {
		cy -= d;
		isChange = true;
	}
	else if (keys[GLFW_KEY_MINUS] &&
		MAX_ITERATION < std::numeric_limits <int>::max() - 10) {
		MAX_ITERATION += 10;
	}
	else if (keys[GLFW_KEY_EQUAL]) {
		MAX_ITERATION -= 10;
		if (MAX_ITERATION <= 0) {
			MAX_ITERATION = 0;
		}
	}

}
static void update_window_title()
{
	std::ostringstream ss;
	ss << "Mandelbrot Renderer";
	ss << ", FPS: " << fps;
	ss << ", Iterations: " << MAX_ITERATION;
	ss << ", Zoom: " << zoom;
	ss << ", At: (" << std::setprecision(8) << cx << " + " << cy << "i)";
	glfwSetWindowTitle(window, ss.str().c_str());
	//isChange = true;
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


void InitShaders(GLuint &prog)
{
	GLint success;
	GLchar infoLog[512];

	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);
	// Check for compile time errors

	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
	// Fragment shader
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);
	// Check for compile time errors
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
	// Link shaders
	prog = glCreateProgram();
	glAttachShader(prog, vertexShader);
	glAttachShader(prog, fragmentShader);
	glLinkProgram(prog);
	// Check for linking errors
	glGetProgramiv(prog, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(prog, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}
float CPU_MandelbrotFunction(double p, double q, int maxIteration)
{
	int i = 0;
	double x_t = 0.0, y_t = 0.0;
	while (i < maxIteration && x_t*x_t + y_t * y_t <= 4.0)
	{
		double temp = x_t * x_t - y_t * y_t + p;
		y_t = 2.0 * x_t * y_t + q;
		x_t = temp;
		i++;
	}

	return (float)i / (float)maxIteration;
}
void setCoord(int x,int y, int offset)
{
	cpu_PixelsMemory[offset * 6] = (-1.0f + 2.0f * (float)(x / (float)WIDTH));			//x
	cpu_PixelsMemory[offset * 6 + 1] = (-1.0f + 2.0f * (float)(y / (float)HEIGHT));		//y
	cpu_PixelsMemory[offset * 6 + 2] = 0.0f;											//z

}
void setColor(float t,int offset)
{
	float r = 9.0 * (1.0 - t) * t * t * t;
	float g = 15.0 * (1.0 - t) * (1.0 - t) * t * t;
	float b = 8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t;

	cpu_PixelsMemory[offset * 6 + 3] = r;			//x
	cpu_PixelsMemory[offset * 6 + 4] = g;			//y
	cpu_PixelsMemory[offset * 6 + 5] = b;			//z
}
void MaldelbrotCPU_Calculation(bool timePrint)
{
	
	clock_t start = clock();
//	cpu_PixelsMemory = (float*)malloc(WIDTH * HEIGHT * sizeof(float) * 6);
	asyncHandler = false;
	int offset;
	double p = 0, q = 0;
	for (int y = 0; y < HEIGHT; y++)
	{
		for (int x = 0; x <WIDTH; x++)
		{

			p = (float)screen_ratio * (float)((float)x / (float)WIDTH - 0.5);
			q = (float)((float)y / (float)HEIGHT - 0.5);
			p /= zoom;
			q /= zoom;
			p += cx;
			q += cy;

			offset = x + y * WIDTH;
			setCoord(x, y, offset);
			setColor(CPU_MandelbrotFunction(p,q, MAX_ITERATION),offset);
		}
	}
	asyncHandler = true;

	if (timePrint)
	{
		clock_t end = clock();
		double seconds = (double)(end - start) / CLOCKS_PER_SEC;
		printf("The time: %f seconds\n", seconds);
		elapsed_time = seconds;
	}
}


// The MAIN function, from here we start the application and run the game loop

int main()
{

	setlocale(LC_ALL, "Russian");
	ofstream time;
	ofstream iters;
	time.open("./time.txt", ios::out | ios::trunc);
	iters.open("./iters.txt", ios::out | ios::trunc);
	//std::thread thread(MaldelbrotCPU_Calculation);
	

	// Init GLFW
	glfwInit();
	// Set all the required options for GLFW
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	atexit(glfwTerminate);


	// Create a GLFWwindow object that we can use for GLFW's functions
	window = glfwCreateWindow(WIDTH, HEIGHT, "MandelbrotCuda", nullptr, nullptr);

	// Set the required callback functions
	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, cursor_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetInputMode(window, GLFW_CURSOR_NORMAL, GLFW_STICKY_KEYS);

	glfwMakeContextCurrent(window);

	// Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
	glewExperimental = GL_TRUE;
	// Initialize GLEW to setup the OpenGL Function pointers
	glewInit();

	// Define the viewport dimensions
	glViewport(0, 0, WIDTH, HEIGHT);


	// Build and compile our shader program
	// Vertex shader



	// Set up vertex data (and buffer(s)) and attribute pointers
	GLfloat vertices[] = {
		// Positions         // Colors
		 0.5f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f,  // Bottom Right
		-0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,  // Bottom Left
		 0.0f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f   // Top
	};


	GLuint VBO, VAO;

	MaldelbrotCPU_Calculation(true);

	InitShaders(shaderProgram);

	glGenVertexArrays(1, &VAO);


	glBindVertexArray(VAO);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);


	glBufferData(GL_ARRAY_BUFFER, sizeof(cpu_PixelsMemory), cpu_PixelsMemory, GL_DYNAMIC_DRAW);

	// Position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	// Color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	glBindVertexArray(0); // Unbind VAO

	last_time = glfwGetTime();
	// Game loop
	while (!glfwWindowShouldClose(window))
	{
		
			MaldelbrotCPU_Calculation(true);
				
				if(asyncHandler)
					auto handle = std::async(std::launch::async,
				MaldelbrotCPU_Calculation,true);
				if (asyncHandler)
					glBufferData(GL_ARRAY_BUFFER, sizeof(cpu_PixelsMemory), cpu_PixelsMemory, GL_DYNAMIC_DRAW);
				
		
		// Check if any events have been activiated (key pressed, mouse moved etc.) and call corresponding response functions
		glfwPollEvents();

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Draw the triangle
		glUseProgram(shaderProgram);
		glBindVertexArray(VAO);
		glDrawArrays(GL_POINTS, 0, HEIGHT * WIDTH);
		glBindVertexArray(0);

		// Swap the screen buffers
		glfwSwapBuffers(window);

		ticks++;
		current_time = glfwGetTime();
		if (current_time - last_time > 1.0) {
			fps = ticks;
			update_window_title();
			last_time = glfwGetTime();
			ticks = 0;
		}
		if (MAX_ITERATION < 200)
		{

			iters << MAX_ITERATION << endl;
			time << elapsed_time << endl;
			MAX_ITERATION++;
		}
	}
	// Properly de-allocate all resources once they've outlived their purpose
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	// Terminate GLFW, clearing any resources allocated by GLFW.
	glfwTerminate();
	return 0;
}

