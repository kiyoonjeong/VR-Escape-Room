// This example is heavily based on the tutorial at https://open.gl

// OpenGL Helpers to reduce the clutter
#include "Helpers.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>
#else
// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>
#endif

// Shortcut to avoid Eigen:: and std:: everywhere, DO NOT USE IN .h
using namespace std;
using namespace Eigen;

// Linear Algebra Library
#include <Eigen/Core>
#include <Eigen/Dense>

// Timer
#include <chrono>

// load
#include <fstream>
#include <iostream>
#include <string>

// VertexBufferObject wrapper
VertexBufferObject VBO_Unit;
VertexBufferObject VBO_Unit_f;
VertexBufferObject VBO_Unit_p;

VertexBufferObject VBO_Bunny;
VertexBufferObject VBO_Bunny_f;
VertexBufferObject VBO_Bunny_p;

VertexBufferObject VBO_Bumpy;
VertexBufferObject VBO_Bumpy_f;
VertexBufferObject VBO_Bumpy_p;

VertexBufferObject VBO_Clock;
VertexBufferObject VBO_Clock_p;
VertexBufferObject VBO_Clock_tex;

VertexBufferObject VBO_BG;
VertexBufferObject VBO_BG_p;
VertexBufferObject VBO_BG_tex;

VertexBufferObject VBO_Sofa;
VertexBufferObject VBO_Sofa_p;
VertexBufferObject VBO_Sofa_tex;

VertexBufferObject VBO_Fire;
VertexBufferObject VBO_Fire_p;
VertexBufferObject VBO_Fire_tex;

VertexBufferObject VBO_Ball;
VertexBufferObject VBO_Ball_p;
VertexBufferObject VBO_Ball_tex;

VertexBufferObject VBO_Stair;
VertexBufferObject VBO_Stair_p;
VertexBufferObject VBO_Stair_tex;


// Contains the vertex positions
MatrixXf V_Unit(3,99999);
MatrixXf V_Unit_f(2,66666);
MatrixXf V_Unit_p(3,99999);

MatrixXf V_Bunny(3,99999);
MatrixXf V_Bunny_f(2,66666);
MatrixXf V_Bunny_p(3,99999);

MatrixXf V_Bumpy(3,99999);
MatrixXf V_Bumpy_f(2,66666);
MatrixXf V_Bumpy_p(3,99999);

MatrixXf V_Temp(3,99999);
MatrixXf V_Temp_p(3,99999);
MatrixXf V_Temp_tex(2,66666);

MatrixXf V_Clock(3,99999);
MatrixXf V_Clock_p(3,99999);
MatrixXf V_Clock_tex(2,66666);

MatrixXf V_Sofa(3,99999);
MatrixXf V_Sofa_p(3,99999);
MatrixXf V_Sofa_tex(2,66666);

MatrixXf V_BG(3,99999);
MatrixXf V_BG_p(3,99999);
MatrixXf V_BG_tex(2,66666);

MatrixXf V_Fire(3,99999);
MatrixXf V_Fire_p(3,99999);
MatrixXf V_Fire_tex(2,66666);

MatrixXf V_Ball(3,99999);
MatrixXf V_Ball_p(3,99999);
MatrixXf V_Ball_tex(2,66666);

MatrixXf V_Stair(3,99999);
MatrixXf V_Stair_p(3,99999);
MatrixXf V_Stair_tex(2,66666);

int v_count = 0;
int clock_count = 0;
int fire_count13 = 0;
int table_count = 0;
int sofa_count = 0;
int ball_count = 0;
int boo_count = 0;


Vector4f Center(0, 0, 0, 1);

int unit_count = 0;
int bunny_count = 0;
int bumpy_count = 0;
const double pi = 3.14159265358979323846;
bool select = false;
int select_i = -1;
float cam_up = 0.0;
float cam_right = 0.0;
Matrix3f RX;
Matrix3f RY;

Matrix4f Morth;
Matrix4f Mper;
Matrix4f Mvp;

float r = 2.0f;
float l = -2.0f;
float t = 2.0f;
float b = -2.0f;
float n = -3.0f;
float f = -40.0f;


//view center
//Vector3f view_center(0.0f, 0.0f, (n+f)/2);

// Camera 
Vector3f eye(0,-3,12);
Vector3f changed_eye(0,-3,12);
int cam_rl = 0.0;
int cam_ud = 0.0;

// View Point
MatrixXf view(4,4);

// Camera point to origin

Matrix4f camera(Vector3f eye, int cam_rl, int cam_ud) {
    //Vector3f CameraDirection = eye.normalized();
    Matrix3f cam_R1(3,3);
    cam_R1 <<
    cos(pi * cam_ud / 180), 0, sin(pi * cam_ud / 180),
    0, 1, 0,
    -sin(pi * cam_ud / 180), 0, cos(pi * cam_ud / 180);

    Matrix3f cam_R2(3,3);
    cam_R2 <<
    1, 0, 0,
    0, cos(pi * cam_rl / 180), -sin(pi * cam_rl / 180),
    0, sin(pi * cam_rl / 180), cos(pi * cam_rl / 180);

    Vector3f temp(0,0,1);
    temp = cam_R1*cam_R2*temp;
    Vector3f CameraDirection = temp.normalized();
    Vector3f Up(0.0f, 1.0f, 0.0f);
    Vector3f CameraRight = Up.cross(CameraDirection).normalized();
    Vector3f CameraUp = CameraDirection.cross(CameraRight);

    MatrixXf cam(4,4);
    cam.col(0) << CameraRight(0), CameraUp(0) , CameraDirection(0), 0;
    cam.col(1) << CameraRight(1), CameraUp(1) , CameraDirection(1), 0;
    cam.col(2) << CameraRight(2), CameraUp(2) , CameraDirection(2), 0;
    cam.col(3) << 0, 0, 0, 1;

    MatrixXf mt(4,4);
    mt.col(0) << 1, 0, 0, 0;
    mt.col(1) << 0, 1, 0, 0;
    mt.col(2) << 0, 0, 1, 0;
    mt.col(3) << -eye(0), -eye(1), -eye(2), 1;
    
    return cam*mt;
}

Matrix4f cam = camera(changed_eye, cam_rl, cam_ud);

// Re-size
Matrix4f re_size(float size){
    Matrix4f mat(4,4);
    mat <<
        size, 0, 0, 0,
        0, size, 0, 0,
        0, 0, size, 0,
        0, 0, 0, 1; 
    
    return mat;
}

// Rotate
Matrix4f rotate(float x, float y, float z) {
    MatrixXf RX(4,4);
    MatrixXf RY(4,4);
    MatrixXf RZ(4,4);

    RX <<
        1, 0, 0, 0,
        0, cos(pi * x / 180), -sin(pi * x / 180), 0,
        0, sin(pi * x / 180), cos(pi * x / 180), 0,
        0, 0, 0, 1;

    RY <<
        cos(pi * y / 180), 0, sin(pi * y / 180), 0,
        0, 1, 0, 0,
        -sin(pi * y / 180), 0, cos(pi * y / 180), 0,
        0, 0, 0, 1;

    RZ <<
        cos(pi * z / 180), -sin(pi * z / 180), 0, 0,
        sin(pi * z / 180), cos(pi * z / 180), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    return RY * RX * RZ;
    
}

Matrix4f translate(float x,float y,float z) {
    MatrixXf mat(4,4);
    mat <<
        1, 0, 0, x,
        0, 1, 0, y,
        0, 0, 1, z,
        0, 0, 0, 1;

    return mat;
}

// Object Data - Type, Translate, Rotate, Scale
struct ObjData {
    int type;
    int number;
    int color;
    float size = 1.0;
    float rx =0.0;
    float ry =0.0;
    float rz =0.0;
    float tx =0.0;
    float ty =0.0;
    float tz =0.0;
};
vector<ObjData> Obj;
vector<ObjData> Obj2;



// Load data
MatrixXf file_load(string file, VectorXi &Order, MatrixXf &Flat, MatrixXf &Vertex_normal, MatrixXf &Phong){
    ifstream object(file);
    string firstline;
    getline(object, firstline);
    MatrixXf RY(3,3);
    RY <<
        cos(pi * 90 / 180), 0, sin(pi * 90 / 180),
        0, 1, 0,
        -sin(pi * 90 / 180), 0, cos(pi * 90 / 180);

    
    //load
    int vertex, triangle, zero;
    object >> vertex >> triangle >> zero;
    double coord_x, coord_y, coord_z;

    //find center and size
    float total_x = 0;
    float total_y = 0;
    float total_z = 0;
    float max = 0;
    float min = 1000000;

    MatrixXf coordinate(3, vertex);
    for (int i = 0; i < vertex; i++){
        object >> coord_x >> coord_y >> coord_z;
        coordinate.col(i) << coord_x, coord_y, coord_z;
        total_x += coord_x;
        total_y += coord_y;
        total_z += coord_z;
    }
    Vector3f center{ total_x / vertex, total_y / vertex, total_z / vertex};

    // Center
    for (int i = 0; i < vertex; i++){
        coordinate.col(i) -= center;
        if (coordinate.col(i)(1) < min){
            min = coordinate.col(i)(1);
        }
        for (int j = 0; j < 3; j++){
            if (abs(coordinate.col(i)(j)) > max) {
                max = abs(coordinate.col(i)(j));
            } 
        }
    }

    // Size
    for (int i = 0; i < vertex; i++){
        coordinate.col(i) /= (max/1.5);
    } 
    // Floor
    min /= (max/1.5);
    for (int i = 0; i < vertex; i++){
        coordinate.col(i)(1) -= (10+min);
    }

    // Rotate
    for (int i = 0; i < vertex; i++){
        coordinate.col(i) = RY * coordinate.col(i);
    }
    // Translation
    for (int i = 0; i < vertex; i++){
        coordinate.col(i)(0) -= 8;
        coordinate.col(i)(2) -= 2;
    }


    Order.resize(3*triangle);
    Flat.resize(2,triangle*3);
    Vertex_normal.resize(3,vertex);
    Phong.resize(3,triangle*3);
    MatrixXf V_file(3, 3*triangle);

    int number_three , point_1, point_2, point_3;
    for (int i=0; i < triangle; i++){
        object >> number_three >> point_1 >> point_2 >> point_3;
        V_file.col(3*i) << coordinate.col(point_1);
        V_file.col(3*i + 1) << coordinate.col(point_2);
        V_file.col(3*i + 2)<< coordinate.col(point_3);
        Order(3*i) = point_1;
        Order(3*i+1) = point_2;
        Order(3*i+2) = point_3;

        Vector3f x1 = V_file.col(3*i + 1) - V_file.col(3*i);
        Vector3f x2 = V_file.col(3*i + 2) - V_file.col(3*i);
        Vector3f normal = x1.cross(x2).normalized();
        Vector3f c = (V_file.col(3*i)+V_file.col(3*i+1)+V_file.col(3*i+2))/3;

        Flat.col(3*i) << 0,0;
        Flat.col(3*i+1) << 0,0;
        Flat.col(3*i+2) << 0,0;
        Vertex_normal.col(point_1) += normal;
        Vertex_normal.col(point_2) += normal;
        Vertex_normal.col(point_3) += normal;
    }
    for (int x=0; x< vertex; x++){
        Vertex_normal.col(x) = Vertex_normal.col(x).normalized();
    }
    for (int x=0; x< triangle*3; x++){
        Phong.col(x) = Vertex_normal.col(Order(x));
    }
    return V_file;
}
// Bunny

VectorXi bun_Order(1);
MatrixXf bun_Flat(2,3);
MatrixXf bun_Vertex_normal(3,3);
MatrixXf bun_Phong(3,3);

MatrixXf bunny = file_load("../data/bunny.off",bun_Order,bun_Flat, bun_Vertex_normal,bun_Phong);
MatrixXf Bunny_Flat = bun_Flat;
MatrixXf Bunny_Phong = bun_Phong;

// // Bumpy

// VectorXi bum_Order(1);
// MatrixXf bum_Flat(3,3);
// MatrixXf bum_Vertex_normal(3,3);
// MatrixXf bum_Phong(3,3);

// MatrixXf bumpy = file_load("../data/bumpy_cube.off",bum_Order,bum_Flat,bum_Vertex_normal,bum_Phong);
// MatrixXf Bumpy_Flat = bum_Flat;
// MatrixXf Bumpy_Phong = bum_Phong;


//Unit Cube
MatrixXf load_cube(MatrixXf &Unit_Flat, MatrixXf &Unit_Vertex_normal, MatrixXf &Unit_Phong){
    MatrixXf CubeData(3,8);
    CubeData <<
        0.5f, 0.5f, 0.5f, 0.5f, -0.5f, -0.5f, -0.5f, -0.5f,
        0.5f, 0.5f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, -0.5f,
        0.5f, -0.5f, 0.5f, -0.5f, 0.5f, -0.5f, 0.5f, -0.5f;

    VectorXi Order(36);
    Order <<
        0,1,2,
        3,2,1,
        1,3,5,
        7,5,3,
        4,5,6,
        7,6,5,
        0,4,2,
        6,2,4,
        0,1,4,
        5,4,1,
        2,6,3,
        7,3,6;

    MatrixXf NormalData(3,6);
    NormalData <<
        1.0f,  0.0f, -1.0f, 0.0f, 0.0f, 0.0f,
        0.0f,  0.0f,  0.0f, 0.0f, 1.0f, -1.0f,
        0.0f, -1.0f,  0.0f, 1.0f, 0.0f, 0.0f;
    
    for (int x = 0; x < 6; x++){
        for (int y = 0; y < 6; y++){
            Unit_Phong.col(6*x + y) = NormalData.col(x);
        }
    }

    
    MatrixXf unitcube(3,36);
    for (int x = 0; x < 36; x++){
        unitcube.col(x) = 0.5 * CubeData.col(Order(x));
    }
    for (int x = 0; x < 12; x++){
        Vector3f x1 = unitcube.col(3*x + 1) - unitcube.col(3*x);
        Vector3f x2 = unitcube.col(3*x + 2) - unitcube.col(3*x);
        Vector3f normal = x1.cross(x2).normalized();
        Vector3f c = (unitcube.col(3*x)+unitcube.col(3*x+1)+unitcube.col(3*x+2))/3;

        Unit_Flat.col(3*x) = normal;
        Unit_Flat.col(3*x+1) = normal;
        Unit_Flat.col(3*x+2) = normal;
        Unit_Vertex_normal.col(Order(3*x)) += normal;
        Unit_Vertex_normal.col(Order(3*x+1)) += normal;
        Unit_Vertex_normal.col(Order(3*x+2)) += normal;
    }

    for (int x=0; x<8; x++){
        Unit_Vertex_normal.col(x) = Unit_Vertex_normal.col(x).normalized();
    }

    return unitcube;
}

MatrixXf Unit_Flat(3,36);
MatrixXf Unit_Vertex_normal(3,8);
MatrixXf Unit_Phong(3,36);
MatrixXf unitcube = load_cube(Unit_Flat,Unit_Vertex_normal,Unit_Phong);


//OBJ file loader

class obj3dmodel
{
    struct face{
        int v1;
        int v2;
        int v3;
    };
    
public:
    void readfile(const char* filename, int &v_count, float rotate, float tx, float ty, float tz, float size, int mode);
};

void obj3dmodel::readfile(const char *filename,int &v_count, float rotate, float tx, float ty, float tz, float size, int mode) 
{
    ifstream fin(filename);
    string s;
    string x;
    string str_v("v");
    string str_vn("vn");
    string str_vt("vt");
    string str_f("f");
    string str_face("FACE");
    
    MatrixXf RY(3,3);
    RY <<
        cos(pi * rotate / 180), 0, sin(pi * rotate / 180),
        0, 1, 0,
        -sin(pi * rotate / 180), 0, cos(pi * rotate / 180);

    int num_zero = 0;
    char dlm;
    int vertex_count = 0;

    std::vector<unsigned int> vertexIndices, uvIndices, normalIndices;
    std::vector<Vector3f> temp_vertices;
    std::vector<Vector2f> temp_uvs;
    std::vector<Vector3f> temp_normals;

    MatrixXf V(3,0);
    MatrixXf UV(2,0);
    MatrixXf N(3,0);

    float total_x = 0;
    float total_y = 0;
    float total_z = 0;
    float max = -10000;
    float min = 100000;

    if(!fin)
        return;
    while(fin>>x)
    {
        if (x == str_v){
            Vector3f vertex;
            fin >> vertex(0) >> vertex(1) >> vertex(2);
            total_x += vertex(0);
            total_y += vertex(1);
            total_z += vertex(2);
            vertex_count += 1;
            temp_vertices.push_back(vertex);
            
        }
        else if (x == str_vt){
            Vector2f uv;
            if (mode == 2){
            fin >> uv(0) >> uv(1);
            temp_uvs.push_back(uv);
            }else{
            fin >> uv(0) >> uv(1) >> num_zero;
            temp_uvs.push_back(uv);
            }
        }
        else if (x == str_vn){
            Vector3f normal;
            fin >> normal(0) >> normal(1) >> normal(2);
            temp_normals.push_back(normal);
        }
        
        else if (x == str_f){
            face f1;
            face f2;
            face f3;

            fin >> f1.v1 >> dlm >> f2.v1 >> dlm >> f3.v1 >> f1.v2 >> dlm >> f2.v2 >> dlm >> f3.v2 >> f1.v3 >> dlm >> f2.v3 >> dlm >> f3.v3 ;

            Vector3f v1 = temp_vertices[f1.v1-1];
            Vector3f v2 = temp_vertices[f1.v2-1];
            Vector3f v3 = temp_vertices[f1.v3-1];

            Vector2f t1 = temp_uvs[f2.v1-1];
            Vector2f t2 = temp_uvs[f2.v2-1];
            Vector2f t3 = temp_uvs[f2.v3-1];

            Vector3f n1 = temp_normals[f3.v1-1];
            Vector3f n2 = temp_normals[f3.v2-1];
            Vector3f n3 = temp_normals[f3.v3-1];

            V_Temp.col(v_count) << v1;
            V_Temp_p.col(v_count) << n1;
            V_Temp_tex.col(v_count) << t1;
            v_count += 1;
            V_Temp.col(v_count) << v2;
            V_Temp_p.col(v_count) << n2;
            V_Temp_tex.col(v_count) << t2;
            v_count += 1;
            V_Temp.col(v_count) << v3;
            V_Temp_p.col(v_count) << n3;
            V_Temp_tex.col(v_count) << t3;
            v_count += 1;
        }
        else{
            getline(fin, s);
        }
    }
    Vector3f vector_center{ total_x / vertex_count, total_y / vertex_count, total_z / vertex_count};

    // Center
    for (int i = 0; i < v_count; i++){
        V_Temp.col(i) -= vector_center;
        if (V_Temp.col(i)(1) < min){
            min = V_Temp.col(i)(1);
        }
        for (int j = 0; j < 3; j++){
            if (abs(V_Temp.col(i)(j)) > max) {
                max = abs(V_Temp.col(i)(j));
            }
        }
    }

    // Rotate
    for (int i = 0; i < v_count; i++){
        V_Temp.col(i) = RY * V_Temp.col(i);
        V_Temp_p.col(i) = RY * V_Temp_p.col(i);
    }

    // Size
    for (int i = 0; i < v_count; i++){
        V_Temp.col(i) /= (max/size);
    }    
    // Floor
    min /= (max/size);
    for (int i = 0; i < v_count; i++){
        V_Temp.col(i)(1) -= (10+min);
    } 
    // Translation
    for (int i = 0; i < v_count; i++){
        V_Temp.col(i)(0) += tx;
        V_Temp.col(i)(1) += ty;
        V_Temp.col(i)(2) += tz;

    } 

}





//Triangle Check

float uvt_tri(Vector3f p1, Vector3f p2, Vector3f p3, Vector4f direction, Vector4f origin){

    Vector3f ray_direction = {direction(0),direction(1),direction(2)};
    Vector3f ray_origin = {origin(0),origin(1),origin(2)};
    
    //Cramer's rule
    double a,b,c,d,e,f,g,h,i,j,k,l,m;
    a = (p1-p2)(0);
    b = (p1-p2)(1);
    c = (p1-p2)(2);
    d = (p1-p3)(0);
    e = (p1-p3)(1);
    f = (p1-p3)(2);
    g = ray_direction(0);
    h = ray_direction(1);
    i = ray_direction(2);
    j = (p1 - ray_origin)(0);
    k = (p1 - ray_origin)(1);
    l = (p1 - ray_origin)(2);
    m = a*(e*i - h*f) + b*(g*f-d*i) + c*(d*h - e*g);
    
    double u,v,t;
    
    u = (j*(e*i-h*f) + k*(g*f-d*i) + l*(d*h - e*g))/m;
    v = (i*(a*k-j*b) + h*(j*c-a*l) + g*(b*l - k*c))/m;
    t = -(f*(a*k-j*b) + e*(j*c-a*l) + d*(b*l - k*c))/m;
    
    if (u+v <= 1 && u >= 0 && v>= 0 && t> 0){
        return t;
    }

    return 99999;
}


//Scale & Rotation

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    // Get the position of the mouse in the window
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    // Get the size of the window
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    
    // Convert screen position to world coordinates
    Eigen::Vector4f p_screen(xpos,height-1-ypos,0,1);
    Eigen::Vector4f p_canonical((p_screen[0]/width)*2-1,(p_screen[1]/height)*2-1,0,1);
    Eigen::Vector4f p_world = view.inverse()*p_canonical;

    Matrix4f id_matrix;
    id_matrix <<
            1, 0, 0, 0,
            0 ,1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    // Update the position of the first vertex if the left button is pressed
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS){
        
        float min = 99999;
        Vector4f ray_end(p_world(0),p_world(1),-1,1.0);
        Vector4f ray_start(p_world(0),p_world(1),1,1.0);
        cam = camera(changed_eye, cam_rl, cam_ud);

        Vector4f center = cam * Vector4f(0.0f, 0.0f, 0.0f, 1.0f);
        ray_end = Mper.inverse() * Morth.inverse() * ray_end;
        ray_end /= ray_end(3);
        ray_start = Mper.inverse() * Morth.inverse() * ray_start;
        ray_start /= ray_start(3);

        for (int i = 0; i < Obj.size(); i++){
            Matrix4f size = re_size(Obj[i].size);
            Matrix4f rotation = rotate(Obj[i].rx, Obj[i].ry, Obj[i].rz);
            Matrix4f transition = translate(Obj[i].tx, Obj[i].ty, Obj[i].tz);
            Vector4f new_ray_end =  rotation.inverse() * size.inverse() * transition.inverse() * cam.inverse() *  ray_end;
            Vector4f ray_origin =  rotation.inverse() * size.inverse() * transition.inverse() * cam.inverse() * ray_start;
            Vector4f ray_direction = (new_ray_end - ray_origin).normalized();

            for (int j=0; j < 12; j++){
                float tri_check = uvt_tri(unitcube.col(3*j),unitcube.col(3*j + 1),unitcube.col(3*j + 2), ray_direction , ray_origin);
                if (tri_check < min){
                    min = tri_check;
                    select_i = i;
                    select = true;

                    break;
                }
            }
        }

        for (int i = 0; i < Obj2.size(); i++){
            Matrix4f size = re_size(Obj2[i].size);
            Matrix4f rotation = rotate(Obj2[i].rx, Obj2[i].ry, Obj2[i].rz);
            Matrix4f transition = translate(Obj2[i].tx, Obj2[i].ty, Obj2[i].tz);
            Vector4f new_ray_end =  rotation.inverse() * size.inverse() * transition.inverse() * cam.inverse() *  ray_end;
            Vector4f ray_origin =  rotation.inverse() * size.inverse() * transition.inverse() * cam.inverse() * ray_start;
            Vector4f ray_direction = (new_ray_end - ray_origin).normalized();

            for (int j=0; j < ball_count/3; j++){
                float tri_check = uvt_tri(V_Ball.col(3*j),V_Ball.col(3*j + 1),V_Ball.col(3*j + 2), ray_direction , ray_origin);
                if (tri_check < min){
                    min = tri_check;
                    select_i = 100*(i+1);
                    select = true;

                    Obj2[i].color = (Obj2[i].color + 1) % 7;

                    break;
                }
            }
        }


        Vector4f new_ray_end = cam.inverse() *  ray_end;
        Vector4f ray_origin = cam.inverse() * ray_start;
        Vector4f ray_direction = (new_ray_end - ray_origin).normalized();
        for (int j=0; j < clock_count/3; j++){
            float tri_check = uvt_tri(V_Clock.col(3*j),V_Clock.col(3*j + 1),V_Clock.col(3*j + 2), ray_direction , ray_origin);
            if (tri_check < min){
                min = tri_check;
                select_i = 10;
                select = true;
            }
        }
        for (int j=0; j < clock_count/3; j++){
            float tri_check = uvt_tri(V_Clock.col(3*j),V_Clock.col(3*j + 1),V_Clock.col(3*j + 2), ray_direction , ray_origin);
            if (tri_check < min){
                min = tri_check;
                select_i = 10;
                select = true;
            }
        }
        for (int j=0; j < sofa_count/3; j++){
            float tri_check = uvt_tri(V_Sofa.col(3*j),V_Sofa.col(3*j + 1),V_Sofa.col(3*j + 2), ray_direction , ray_origin);
            if (tri_check < min){
                min = tri_check;
                select_i = 20;
                select = true;
            }
        }
        for (int j=0; j < table_count/3; j++){
            float tri_check = uvt_tri(V_Bumpy.col(3*j),V_Bumpy.col(3*j + 1),V_Bumpy.col(3*j + 2), ray_direction , ray_origin);
            if (tri_check < min){
                min = tri_check;
                select_i = 30;
                select = true;
            }
        }
        for (int j=0; j < fire_count13/3; j++){
            float tri_check = uvt_tri(V_Fire.col(3*j),V_Fire.col(3*j + 1),V_Fire.col(3*j + 2), ray_direction , ray_origin);
            if (tri_check < min){
                min = tri_check;
                select_i = 40;
                select = true;
            }
        }
        for (int j=0; j < ball_count/3; j++){
            float tri_check = uvt_tri(V_Ball.col(3*j),V_Ball.col(3*j + 1),V_Ball.col(3*j + 2), ray_direction , ray_origin);
            if (tri_check < min){
                min = tri_check;
                select_i = 50;
                select = true;
            }
        }
        for (int j=0; j < 1000; j++){
            float tri_check = uvt_tri(V_Bunny.col(3*j),V_Bunny.col(3*j + 1),V_Bunny.col(3*j + 2), ray_direction , ray_origin);
            if (tri_check < min){
                min = tri_check;
                select_i = 60;
                select = true;
            }
        }
        
        if (min == 99999){
            select_i = -1;
            select = false;
        }
        std::cout << "Select : " << select_i << std::endl;
    }
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    // Update the position of the first vertex if the keys 1,2, or 3 are pressed
    switch (key)
    {        
        case GLFW_KEY_O:
            //size up
            if (action == GLFW_PRESS && select){
                if (Obj[select_i].size < 4.0){
                    Obj[select_i].size += 0.2;
                }
            }
            break;

        case GLFW_KEY_L:
            //size down
            if (action == GLFW_PRESS && select){
                if (Obj[select_i].size > 1.0){
                    Obj[select_i].size -= 0.2;
                }
            }
            break; 

        case GLFW_KEY_H:
            //rotate
            if (action == GLFW_PRESS && select){
                Obj[select_i].rx += 10;
            }
            break; 
        
        case GLFW_KEY_Y:
            //rotate
            if (action == GLFW_PRESS && select){
                Obj[select_i].rx -= 10;
            }
            break; 

        case GLFW_KEY_J:
            //rotate
            if (action == GLFW_PRESS && select){
                Obj[select_i].ry += 10;
            }
            break; 
        
        case GLFW_KEY_U:
            //rotate
            if (action == GLFW_PRESS && select){
                Obj[select_i].ry -= 10;
            }
            break; 

        case GLFW_KEY_K:
            //rotate
            if (action == GLFW_PRESS && select){
                Obj[select_i].rz += 10;
            }
            break; 

        case GLFW_KEY_I:
            //rotate
            if (action == GLFW_PRESS && select){
                Obj[select_i].rz -= 10;
            }
            break; 

        case GLFW_KEY_D:
            //translate
            if (action == GLFW_PRESS && select){
                Obj[select_i].tx += 0.2;
            }
            break; 
        
        case GLFW_KEY_E:
            //translate
            if (action == GLFW_PRESS && select){
                Obj[select_i].tx -= 0.2;
            }
            break; 

        case GLFW_KEY_F:
            //translate
            if (action == GLFW_PRESS && select){
                Obj[select_i].ty += 0.2;
            }
            break; 
        
        case GLFW_KEY_R:
            //translate
            if (action == GLFW_PRESS && select){
                Obj[select_i].ty -= 0.2;
            }
            break; 

        case GLFW_KEY_G:
            //translate
            if (action == GLFW_PRESS && select){
                Obj[select_i].tz += 0.2;
            }
            break; 

        case GLFW_KEY_T:
            //translate
            if (action == GLFW_PRESS && select){
                Obj[select_i].tz -= 0.2;
            }
            break;


        case  GLFW_KEY_1:
            //move right
            if (action == GLFW_PRESS){ 
                changed_eye(0) = min(6.0, changed_eye(0) + 0.2);
                select_i = -1;
                select = false;
            }
            break;
            
        case  GLFW_KEY_2:
            // move left
            if (action == GLFW_PRESS){ 
                changed_eye(0) = max(-6.0, changed_eye(0) - 0.2);
                select_i = -1;
                select = false;
            }
            break;

        case  GLFW_KEY_3:
            // stand up
            if (action == GLFW_PRESS){ 
                changed_eye(1) = -3;
                select_i = -1;
                select = false;
            }
            break;
        
        case  GLFW_KEY_4:
            // sit down
            if (action == GLFW_PRESS){ 
                changed_eye(1) = -6;
                select_i = -1;
                select = false;
            }
            break;

        case GLFW_KEY_5:
            // eye far
            if (action == GLFW_PRESS){
                changed_eye(2) = min(14.0, changed_eye(2) + 0.2);
                select_i = -1;
                select = false;
            }
            break;

        case GLFW_KEY_6:
            // eye close
            if (action == GLFW_PRESS){
                changed_eye(2) = max(-3.0, changed_eye(2) - 0.2);
                select_i = -1;
                select = false;
            }
            break;

        case GLFW_KEY_7:
            // camera up
            if (action == GLFW_PRESS){
                cam_ud = min(70,cam_ud+10);
                select_i = -1;
                select = false;
            }
            break;

        case GLFW_KEY_8:
            // camera down
            if (action == GLFW_PRESS){
                cam_ud = max(-70, cam_ud-10);
                select_i = -1;
                select = false;
            }
            break;

        case GLFW_KEY_9:
            // camera right
            if (action == GLFW_PRESS){
                cam_rl = min(70, cam_rl+10);
                select_i = -1;
                select = false;
            }
            break;

        case GLFW_KEY_0:
            // camera left
            if (action == GLFW_PRESS){
                cam_rl = max(-70, cam_rl-10);
                select_i = -1;
                select = false;

            }
            break;

        default:
            break;
    }
}

int main(void)
{
    GLFWwindow* window;

    // Initialize the library
    if (!glfwInit())
        return -1;

    // Activate supersampling
    glfwWindowHint(GLFW_SAMPLES, 8);

    // Ensure that we get at least a 3.2 context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

    // On apple we have to load a core profile with forward compatibility
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(800, 600, "VR ESCAPE ROOM", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    #ifndef __APPLE__
      glewExperimental = true;
      GLenum err = glewInit();
      if(GLEW_OK != err)
      {
        /* Problem: glewInit failed, something is seriously wrong. */
       fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
      }
      glGetError(); // pull and savely ignonre unhandled errors like GL_INVALID_ENUM
      fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
    #endif

    int major, minor, rev;
    major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
    minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
    rev = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
    printf("OpenGL version recieved: %d.%d.%d\n", major, minor, rev);
    printf("Supported OpenGL is %s\n", (const char*)glGetString(GL_VERSION));
    printf("Supported GLSL is %s\n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

    // Initialize the VAO
    // A Vertex Array Object (or VAO) is an object that describes how the vertex
    // attributes are stored in a Vertex Buffer Object (or VBO). This means that
    // the VAO is not the actual object storing the vertex data,
    // but the descriptor of the vertex data.
    VertexArrayObject VAO;
    VAO.init();
    VAO.bind();

    // Initialize the VBO with the vertices data
    // A VBO is a data container that lives in the GPU memory
    VBO_Unit.init();
    VBO_Unit_f.init();
    VBO_Unit_p.init();

    VBO_Bunny.init();
    VBO_Bunny_f.init();
    VBO_Bunny_p.init();
    
    VBO_Bumpy.init();
    VBO_Bumpy_f.init();
    VBO_Bumpy_p.init();

    VBO_Clock.init();
    VBO_Clock_p.init();
    VBO_Clock_tex.init();

    VBO_Sofa.init();
    VBO_Sofa_p.init();
    VBO_Sofa_tex.init();

    VBO_BG.init();
    VBO_BG_p.init();
    VBO_BG_tex.init();

    VBO_Fire.init();
    VBO_Fire_p.init();
    VBO_Fire_tex.init();

    VBO_Ball.init();
    VBO_Ball_p.init();
    VBO_Ball_tex.init();

    VBO_Stair.init();
    VBO_Stair_p.init();
    VBO_Stair_tex.init();

    VBO_Unit.update(V_Unit);
    VBO_Unit_f.update(V_Unit_f);
    VBO_Unit_p.update(V_Unit_p);

    VBO_Bunny.update(V_Bunny);
    VBO_Bunny_f.update(V_Bunny_f);
    VBO_Bunny_p.update(V_Bunny_p);

    VBO_Bumpy.update(V_Bumpy);
    VBO_Bumpy_f.update(V_Bumpy_f);
    VBO_Bumpy_p.update(V_Bumpy_p);

    VBO_Clock.update(V_Clock);
    VBO_Clock_p.update(V_Clock_p);
    VBO_Clock_tex.update(V_Clock_tex);

    VBO_Sofa.update(V_Sofa);
    VBO_Sofa_p.update(V_Sofa_p);
    VBO_Sofa_tex.update(V_Sofa_tex);

    VBO_Fire.update(V_Fire);
    VBO_Fire_p.update(V_Fire_p);
    VBO_Fire_tex.update(V_Fire_tex);

    VBO_Ball.update(V_Ball);
    VBO_Ball_p.update(V_Ball_p);
    VBO_Ball_tex.update(V_Ball_tex);

    VBO_Stair.update(V_Stair);
    VBO_Stair_p.update(V_Stair_p);
    VBO_Stair_tex.update(V_Stair_tex);

    Morth <<
        2/(r-l), 0, 0, -(r+l)/(r-l),
        0, 2/(t-b), 0, -(t+b)/(t-b),
        0, 0, 2/(n-f), -(n+f)/(n-f),
        0, 0, 0, 1;

    Mper <<
        n, 0, 0, 0,
        0, n, 0, 0,
        0, 0, n+f, -f*n,
        0, 0, 1, 0;


    RX <<
    1, 0, 0,
    0, 1, 0, 
    0, 0, 1;

    RY <<
    1, 0, 0,
    0, 1, 0, 
    0, 0, 1;







    // *******************************************************

    obj3dmodel obj3d = obj3dmodel();
    obj3d.readfile("../data/longcase_clock.obj", v_count, 0.0, 8.0, 0.0, -8, 5, 3 );

    V_Clock = V_Temp;
    V_Clock_p = V_Temp_p;
    V_Clock_tex = V_Temp_tex;
    clock_count = v_count;
    // std::cout << "count : " << v_count << std::endl;
    // std::cout << "V_Clockf: " << V_Clock.col(v_count-1) << std::endl; 
    v_count = 0;

    obj3d.readfile("../data/sofa4.obj", v_count, 270.0, 8.0, 0.0, 0.0, 5,3);
    V_Sofa = V_Temp;
    V_Sofa_p = V_Temp_p;
    V_Sofa_tex = V_Temp_tex;
    sofa_count = v_count;
    v_count = 0;
    
    obj3d.readfile("../data/table_half_round.obj", v_count, 0.0,0.0,0.0,0.0,2,3);
    V_Bumpy = V_Temp;
    V_Bumpy_p = V_Temp_p;
    V_Bumpy_f = V_Temp_tex;
    table_count = v_count; 
    v_count = 0;

    obj3d.readfile("../data/FP2015.obj", v_count, 0.0,0.0,0.0,-8,4,2);
    V_Fire = V_Temp;
    V_Fire_p = V_Temp_p;
    V_Fire_tex = V_Temp_tex;
    int fire_count1 = 48 * 3;
    int fire_count2 = 72 * 3;
    int fire_count3 = 120 * 3;
    int fire_count4 = 144 * 3;
    int fire_count5 = 168 * 3;
    int fire_count6 = 192 * 3;
    int fire_count7 = 216 * 3;
    int fire_count8 = 240 * 3;
    int fire_count9 = 264 * 3;
    int fire_count10 = 288 * 3;
    int fire_count11 = 312 * 3;
    int fire_count12 = 336 * 3;
    fire_count13 = 360 * 3;
    v_count = 0;
    

    obj3d.readfile("../data/fireball.obj", v_count, 0.0,0.0,9.5,0.0,0.5,2);
    V_Ball = V_Temp;
    V_Ball_p = V_Temp_p;
    V_Ball_tex = V_Temp_tex;
    ball_count = v_count; 
    v_count = 0;

    obj3d.readfile("../data/stair.obj", v_count, 0.0, -8.0, 0.0, 0.0, 10,2);
    V_Stair = V_Temp;
    V_Stair_p = V_Temp_p;
    V_Stair_tex = V_Temp_tex;
    int stair_count = v_count;
    v_count = 0;

    //Bunny
    V_Bunny = bunny;
    V_Bunny_f = Bunny_Flat;
    V_Bunny_p = Bunny_Phong;

    // Floor
    MatrixXf floor_V(3,3600);
    MatrixXf floor_normal(3,3600);
    MatrixXf floor_tex(2,3600);
    for (int i=0 ; i < 10 ; i++){
        for (int j=0 ; j < 10 ; j++){

            // floor
            floor_V.col(36*(10*i+j)) << -10 + 2*i , -10, -10 + 3*j;
            floor_V.col(36*(10*i+j)+1) << -10 + 2*i , -10, -10 + 3*(j+1);
            floor_V.col(36*(10*i+j)+2) << -10 + 2*(i+1) , -10, -10 + 3*(j+1);
            floor_V.col(36*(10*i+j)+3) << -10 + 2*(i+1) , -10, -10 + 3*(j+1);
            floor_V.col(36*(10*i+j)+4) << -10 + 2*(i+1) , -10, -10 + 3*j;
            floor_V.col(36*(10*i+j)+5) << -10 + 2*i , -10, -10 + 3*j;

            // Ceiling
            floor_V.col(36*(10*i+j)+6) << -10 + 2*i , 10, -10 + 3*j;
            floor_V.col(36*(10*i+j)+7) << -10 + 2*i , 10, -10 + 3*(j+1);
            floor_V.col(36*(10*i+j)+8) << -10 + 2*(i+1) , 10, -10 + 3*(j+1);
            floor_V.col(36*(10*i+j)+9) << -10 + 2*(i+1) , 10, -10 + 3*(j+1);
            floor_V.col(36*(10*i+j)+10) << -10 + 2*(i+1) , 10, -10 + 3*j;
            floor_V.col(36*(10*i+j)+11) << -10 + 2*i , 10, -10 + 3*j;

            // Wall.F
            floor_V.col(36*(10*i+j)+12) << -10 + 2*i , -10 + 2*j, -10;
            floor_V.col(36*(10*i+j)+13) << -10 + 2*i , -10 + 2*(j+1), -10;
            floor_V.col(36*(10*i+j)+14) << -10 + 2*(i+1) , -10 + 2*(j+1), -10;
            floor_V.col(36*(10*i+j)+15) << -10 + 2*(i+1) , -10 + 2*(j+1), -10;
            floor_V.col(36*(10*i+j)+16) << -10 + 2*(i+1) , -10 + 2*j, -10;
            floor_V.col(36*(10*i+j)+17) << -10 + 2*i , -10 + 2*j, -10;

            // Wall.B
            floor_V.col(36*(10*i+j)+18) << -10 + 2*(i+1) , -10 + 2*j, 20;
            floor_V.col(36*(10*i+j)+19) << -10 + 2*(i+1) , -10 + 2*(j+1), 20;
            floor_V.col(36*(10*i+j)+20) << -10 + 2*i , -10 + 2*(j+1), 20;
            floor_V.col(36*(10*i+j)+21) << -10 + 2*i , -10 + 2*(j+1), 20;
            floor_V.col(36*(10*i+j)+22) << -10 + 2*i , -10 + 2*j, 20;
            floor_V.col(36*(10*i+j)+23) << -10 + 2*(i+1) , -10 + 2*j, 20;

            // Wall.R
            floor_V.col(36*(10*i+j)+24) << 10 , -10 + 2*i, -10 + 3*j;
            floor_V.col(36*(10*i+j)+25) << 10 , -10 + 2*i, -10 + 3*(j+1);
            floor_V.col(36*(10*i+j)+26) << 10 , -10 + 2*(i+1), -10 + 3*(j+1);
            floor_V.col(36*(10*i+j)+27) << 10 , -10 + 2*(i+1), -10 + 3*(j+1);
            floor_V.col(36*(10*i+j)+28) << 10 , -10 + 2*(i+1), -10 + 3*j;
            floor_V.col(36*(10*i+j)+29) << 10 , -10 + 2*i, -10 + 3*j;

            // Wall.L
            floor_V.col(36*(10*i+j)+30) << -10 , -10 + 2*i, -10 + 3*j;
            floor_V.col(36*(10*i+j)+31) << -10 , -10 + 2*i, -10 + 3*(j+1);
            floor_V.col(36*(10*i+j)+32) << -10 , -10 + 2*(i+1), -10 + 3*(j+1);
            floor_V.col(36*(10*i+j)+33) << -10 , -10 + 2*(i+1), -10 + 3*(j+1);
            floor_V.col(36*(10*i+j)+34) << -10 , -10 + 2*(i+1), -10 + 3*j;
            floor_V.col(36*(10*i+j)+35) << -10 , -10 + 2*i, -10 + 3*j;

            floor_normal.col(36*(10*i+j)) << 0,1,0;
            floor_normal.col(36*(10*i+j)+1) << 0,1,0;
            floor_normal.col(36*(10*i+j)+2) << 0,1,0;
            floor_normal.col(36*(10*i+j)+3) << 0,1,0;
            floor_normal.col(36*(10*i+j)+4) << 0,1,0;
            floor_normal.col(36*(10*i+j)+5) << 0,1,0;
            floor_normal.col(36*(10*i+j)+6) << 0,-1,0;
            floor_normal.col(36*(10*i+j)+7) << 0,-1,0;
            floor_normal.col(36*(10*i+j)+8) << 0,-1,0;
            floor_normal.col(36*(10*i+j)+9) << 0,-1,0;
            floor_normal.col(36*(10*i+j)+10) << 0,-1,0;
            floor_normal.col(36*(10*i+j)+11) << 0,-1,0;

            // f
            floor_normal.col(36*(10*i+j)+12) << 0,0,1;
            floor_normal.col(36*(10*i+j)+13) << 0,0,1;
            floor_normal.col(36*(10*i+j)+14) << 0,0,1;
            floor_normal.col(36*(10*i+j)+15) << 0,0,1;
            floor_normal.col(36*(10*i+j)+16) << 0,0,1;
            floor_normal.col(36*(10*i+j)+17) << 0,0,1;

            // b
            floor_normal.col(36*(10*i+j)+18) << 0,0,-1;
            floor_normal.col(36*(10*i+j)+19) << 0,0,-1;
            floor_normal.col(36*(10*i+j)+20) << 0,0,-1;
            floor_normal.col(36*(10*i+j)+21) << 0,0,-1;
            floor_normal.col(36*(10*i+j)+22) << 0,0,-1;
            floor_normal.col(36*(10*i+j)+23) << 0,0,-1;

            // R
            floor_normal.col(36*(10*i+j)+24) << -1,0,0;
            floor_normal.col(36*(10*i+j)+25) << -1,0,0;
            floor_normal.col(36*(10*i+j)+26) << -1,0,0;
            floor_normal.col(36*(10*i+j)+27) << -1,0,0;
            floor_normal.col(36*(10*i+j)+28) << -1,0,0;
            floor_normal.col(36*(10*i+j)+29) << -1,0,0;

            // L
            floor_normal.col(36*(10*i+j)+30) << 1,0,0;
            floor_normal.col(36*(10*i+j)+31) << 1,0,0;
            floor_normal.col(36*(10*i+j)+32) << 1,0,0;
            floor_normal.col(36*(10*i+j)+33) << 1,0,0;
            floor_normal.col(36*(10*i+j)+34) << 1,0,0;
            floor_normal.col(36*(10*i+j)+35) << 1,0,0;

            for (int k=0; k <6 ;k++ ){
                // floor_tex.col(36*(10*i+j)+6*k) << 0,0;
                // floor_tex.col(36*(10*i+j)+6*k+1) << 0,1;
                // floor_tex.col(36*(10*i+j)+6*k+2) << 1,1;
                // floor_tex.col(36*(10*i+j)+6*k+3) << 1,1;
                // floor_tex.col(36*(10*i+j)+6*k+4) << 1,0;
                // floor_tex.col(36*(10*i+j)+6*k+5) << 0,0;
                floor_tex.col(36*(10*i+j)+6*k) << 0.1*i,0.1*j;
                floor_tex.col(36*(10*i+j)+6*k+1) << 0.1*i,0.1*(j+1);
                floor_tex.col(36*(10*i+j)+6*k+2) << 0.1*(i+1),0.1*(j+1);
                floor_tex.col(36*(10*i+j)+6*k+3) << 0.1*(i+1),0.1*(j+1);
                floor_tex.col(36*(10*i+j)+6*k+4) << 0.1*(i+1),0.1*j;
                floor_tex.col(36*(10*i+j)+6*k+5) << 0.1*i,0.1*j;
            }

        }    
    }

    //cube
    for (int i=0; i <7 ;i++ ){
        struct ObjData unit_data;
        unit_data.type = 1;
        unit_data.number = unit_count;
        Obj.push_back(unit_data);
        unit_count += 1;
    }

    MatrixXf cube_tex(2,36);
    for (int i=0; i <6 ;i++ ){
        cube_tex.col(6*i) << 1,1;
        cube_tex.col(6*i+1) << 0, 1;
        cube_tex.col(6*i+2) << 1,0;
        cube_tex.col(6*i+3) << 0,0;
        cube_tex.col(6*i+4) << 1,0;
        cube_tex.col(6*i+5) << 0,1;
    }

    MatrixXf zero_tex(2,36);
    for (int i=0; i <6 ;i++ ){
        zero_tex.col(6*i) << 0,0;
        zero_tex.col(6*i+1) << 0,0;
        zero_tex.col(6*i+2) << 0,0;
        zero_tex.col(6*i+3) << 0,0;
        zero_tex.col(6*i+4) << 0,0;
        zero_tex.col(6*i+5) << 0,0;
    }

    MatrixXf cube_tex2(2,36);
    for (int i=0; i <6 ;i++ ){
        cube_tex2.col(6*i) << 0, 1;
        cube_tex2.col(6*i+1) << 1, 1;
        cube_tex2.col(6*i+2) << 0, 0;
        cube_tex2.col(6*i+3) << 1, 0;
        cube_tex2.col(6*i+4) << 0, 0;
        cube_tex2.col(6*i+5) << 1, 1;
    }

    // Cube Locate

    //below table
    Obj[0].type = 2;
    Obj[0].ry = 90;
    Obj[0].tx = 0;
    Obj[0].ty = -9.5;
    Obj[0].tz = -4;
    Obj[0].color = 1;

    //above table
    Obj[1].type = 2;
    Obj[1].ry = 180;
    Obj[1].tx = 0;
    Obj[1].ty = -7;
    Obj[1].tz = 0;
    Obj[1].color = 2;

    //above sofa
    Obj[2].type = 2;
    Obj[2].ry = 270;
    Obj[2].tx = 8;
    Obj[2].ty = -7.2;
    Obj[2].tz = -1;
    Obj[2].color = 3;

    //right side of sofa
    Obj[3].type = 2;
    Obj[3].ry = 30;
    Obj[3].tx = 8;
    Obj[3].ty = -9.5;
    Obj[3].tz = 8.5;
    Obj[3].color = 4;

    // left side of sofa
    Obj[4].type = 2;
    Obj[4].ry = 60;
    Obj[4].tx = 8.5;
    Obj[4].ty = -9.5;
    Obj[4].tz = -7;
    Obj[4].color = 5;

    // inside the fire place
    Obj[5].type = 2;
    Obj[5].ry = 120;
    Obj[5].tx = 0;
    Obj[5].ty = -9.5;
    Obj[5].tz = -8.5;
    Obj[5].color = 6;

    // ceiling
    Obj[6].type = 2;
    Obj[6].ry = 150;
    Obj[6].tx = 0;
    Obj[6].ty = 9.5;
    Obj[6].tz = 5;
    Obj[6].color = 7;
    
    // Ball locate

    int ball_number = 0;

    for (int i=0; i <7 ;i++ ){
        struct ObjData ball_data;
        ball_data.type = 1;

        //ball count != ball number

        ball_data.number = ball_number;
        Obj2.push_back(ball_data);

        Obj2[i].type = 2;
        Obj2[i].ry = 90;
        Obj2[i].tx = -3+i;
        Obj2[i].ty = 8;
        Obj2[i].tz = 0;
        Obj2[i].color = 7;
        
        ball_number += 1;
    }
    

    

 // *******************************************************
    

   
    //Point check
    Vector4f Checker(1, 1, 1, 1);
    std::cout << "point : " << Checker << std::endl;
    std::cout << "cam : " << cam * Checker << std::endl;
    std::cout << "morph : " << Morth * cam * Checker << std::endl;
    std::cout << "view : " << view * Morth * cam * Checker << std::endl;

    Vector4f check_point1(0,-5,10,1);   
    Vector4f check_point2(0,-5,5,1);
    Vector4f check_point3(0, -10,5,1);
    Vector4f check_point4(0, -7.5,5,1);
    Vector4f check_point5(0,-5,10,1);

    std::cout << "check1 : " <<  Morth * Mper * cam * check_point1 << std::endl;
    std::cout << "check2 : " <<  Morth * Mper * cam * check_point2 << std::endl;
    std::cout << "check3 : " <<  Morth * Mper * cam * check_point3 << std::endl;
    std::cout << "check4 : " <<  Morth * Mper * cam * check_point4 << std::endl;
    std::cout << "check5 : " <<  Morth * Mper * cam * check_point5 << std::endl;


    // Initialize the OpenGL Program
    // A program controls the OpenGL pipeline and it must contains
    // at least a vertex shader and a fragment shader to be valid
    Program program;
    const GLchar* vertex_shader =
            "#version 150 core\n"
                    "in vec2 texcoord;"
                    "in vec3 position;"
                    "uniform vec3 light;"
                    "uniform vec3 eye;"
                    "in vec3 normal;"
                    "uniform vec3 triangleColor;"
                    "uniform mat4 view;"
                    "uniform mat4 Morth;"
                    "uniform mat4 Mper;"
                    "uniform mat4 cam;"
                    "uniform mat4 size;"
                    "uniform mat4 rotation;"
                    "uniform mat4 transition;"
                    "out vec3 f_color;"
                    "out vec2 Texcoord;"
                    "out float w_value;"

                    "void main()"
                    "{"
                    "   vec4 temp = cam * transition * size * rotation * vec4(position, 1.0);"
                    "   vec4 final_position = Morth * Mper * temp;"
                    "   gl_Position = view * (final_position / -abs(final_position.w));"


                    "   vec4 location =  transition * size * rotation * vec4(position, 1.0);" 
                    "   vec4 l_direction = normalize(vec4(light, 1.0) - location);"
                    "   vec4 v_direction = normalize(vec4(eye, 1.0) - location);"
                    "   vec4 h = normalize(v_direction + l_direction);"
                    "   float value1 = max(dot(normalize(transpose(inverse(rotation)) * vec4(normal, 0.0)),l_direction), 0.0f);"
                    "   float value2 = pow(max(dot(normalize(transpose(inverse(rotation)) * vec4(normal, 0.0)), h), 0.0f),32);"
                    "   f_color = (value1 + 0.5*value2 + 0.1) * triangleColor;"
                    //"   Texcoord = texcoord/final_position.w;"
                    "   Texcoord = texcoord;"
                    //"   w_value = final_position.w;"
                    "}";
    const GLchar* fragment_shader =
            "#version 150 core\n"
                    "in vec3 f_color;"
                    "in vec2 Texcoord;"
                    //"in float w_value;"

                    "out vec4 outColor;"

                    "uniform sampler2D tex;"

                    "void main()"
                    "{"
                    //"    outColor = vec4(texture(tex, w_value * Texcoord.xy).rgb, 1.0) * vec4(f_color, 1.0);"
                    "    outColor = vec4(texture(tex, Texcoord.xy).rgb, 1.0) * vec4(f_color, 1.0);"
                    "}";

    // Compile the two shaders and upload the binary to the GPU
    // Note that we have to explicitly specify that the output "slot" called outColor
    // is the one that we want in the fragment buffer (and thus on screen)
    program.init(vertex_shader,fragment_shader,"outColor");
    program.bind();

    // The vertex shader wants the position of the vertices as an input.
    // The following line connects the VBO we defined above with the position "slot"
    // in the vertex shader

    // Save the current time --- it will be used to dynamically change the triangle color
    auto t_start = std::chrono::high_resolution_clock::now();

    // Register the keyboard callback
    glfwSetKeyCallback(window, key_callback);

    // Register the mouse callback
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // Update viewport
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // load texture
    unsigned int tex1, tex2, tex3, tex4, tex5, tex6, tex7, tex8, tex9;
    int texture_width, texture_height, bpp;
    unsigned char *rgb_array;

    
    glGenTextures(1, &tex1);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    rgb_array = stbi_load("../data/texture0.png", &texture_width, &texture_height, &bpp, 3);
    stbi_set_flip_vertically_on_load(true);
    if(rgb_array == nullptr)
        printf("Cannot load texture image.\n");
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_width, texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_array);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(rgb_array);

    glGenTextures(1, &tex2);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex2);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    rgb_array = stbi_load("../data/texture2.png", &texture_width, &texture_height, &bpp, 3);
    stbi_set_flip_vertically_on_load(true);
    if(rgb_array == nullptr)
        printf("Cannot load texture image.\n");
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_width, texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_array);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(rgb_array);


    glGenTextures(1, &tex3);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex3);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    rgb_array = stbi_load("../data/texture3.png", &texture_width, &texture_height, &bpp, 3);
    stbi_set_flip_vertically_on_load(true);
    if(rgb_array == nullptr)
        printf("Cannot load texture image.\n");
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_width, texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_array);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(rgb_array);

    glGenTextures(1, &tex4);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex4);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    rgb_array = stbi_load("../data/texture4.png", &texture_width, &texture_height, &bpp, 3);
    stbi_set_flip_vertically_on_load(true);
    if(rgb_array == nullptr)
        printf("Cannot load texture image.\n");
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_width, texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_array);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(rgb_array);

    glGenTextures(1, &tex5);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, tex5);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    rgb_array = stbi_load("../data/texture5.png", &texture_width, &texture_height, &bpp, 3);
    stbi_set_flip_vertically_on_load(true);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_width, texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_array);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(rgb_array);

    glGenTextures(1, &tex6);
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, tex6);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    rgb_array = stbi_load("../data/texture6.jpg", &texture_width, &texture_height, &bpp, 3);
    stbi_set_flip_vertically_on_load(true);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_width, texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_array);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(rgb_array);

    glGenTextures(1, &tex7);
    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_2D, tex7);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    rgb_array = stbi_load("../data/texture7.png", &texture_width, &texture_height, &bpp, 3);
    stbi_set_flip_vertically_on_load(true);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_width, texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_array);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(rgb_array);

    glGenTextures(1, &tex8);
    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_2D, tex8);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    rgb_array = stbi_load("../data/texture8.png", &texture_width, &texture_height, &bpp, 3);
    stbi_set_flip_vertically_on_load(true);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_width, texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_array);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(rgb_array);

    glGenTextures(1, &tex9);
    glActiveTexture(GL_TEXTURE8);
    glBindTexture(GL_TEXTURE_2D, tex9);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    rgb_array = stbi_load("../data/texture9.png", &texture_width, &texture_height, &bpp, 3);
    stbi_set_flip_vertically_on_load(true);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_width, texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_array);
    glGenerateMipmap(GL_TEXTURE_2D);
    stbi_image_free(rgb_array);


    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window))
    {
        // Bind your VAO (not necessary if you have only one)
        VAO.bind();

        // Bind your program
        program.bind();

        // depth
        glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
        glEnable(GL_DEPTH_TEST);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        glClearDepth(0.0f);
        glDepthFunc(GL_GREATER);

        // Camera

        Matrix4f cam = camera(changed_eye, cam_rl, cam_ud);

        // Get size of the window
        int width, height;
        glfwGetWindowSize(window, &width, &height);

        float aspect_ratio = float(height)/float(width);

        view <<
        1 ,0, 0, 0,
        0,          1, 0, 0,
        0,           0, 1, 0,
        0,           0, 0, 1;

        glUniformMatrix4fv(program.uniform("view"), 1, GL_FALSE, view.data());
        glUniformMatrix4fv(program.uniform("cam"), 1, GL_FALSE, cam.data());
		glUniformMatrix4fv(program.uniform("Morth"), 1, GL_FALSE, Morth.data());
        glUniformMatrix4fv(program.uniform("Mper"), 1, GL_FALSE, Mper.data());
        glUniform3f(program.uniform("light"), 0.0f, 0.0f, 0.0f);
        glUniform3f(program.uniform("eye"), changed_eye(0), changed_eye(1), changed_eye(2));

        // Clear the framebuffer
        glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
    
        
        // Draw a clock

        Matrix4f size;
        size <<
                1, 0, 0, 0,
                0 ,1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;
        Matrix4f rotation;
        rotation <<
                1, 0, 0, 0,
                0 ,1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;
        Matrix4f transition;
        transition <<
                1, 0, 0, 0,
                0 ,1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;
        
        // Clock
        VBO_Clock.update(V_Clock);
        VBO_Clock_p.update(V_Clock_p);
        VBO_Clock_tex.update(V_Clock_tex);
        glUniformMatrix4fv(program.uniform("size"), 1, GL_FALSE, size.data());
        glUniformMatrix4fv(program.uniform("rotation"), 1, GL_FALSE, rotation.data());
        glUniformMatrix4fv(program.uniform("transition"), 1, GL_FALSE, transition.data());
        
        glUniform1i(program.uniform("tex"), 3);
        program.bindVertexAttribArray("position",VBO_Clock);
        program.bindVertexAttribArray("normal",VBO_Clock_p);
        program.bindVertexAttribArray("texcoord",VBO_Clock_tex);
        glUniform3f(program.uniform("triangleColor"), 1.0f, 1.0f, 1.0f);
        glDrawArrays(GL_TRIANGLES, 0, clock_count-12);
        glUniform1i(program.uniform("tex"), 1);
        glDrawArrays(GL_TRIANGLES, clock_count-12, 12);

        //Sofa
        VBO_Sofa.update(V_Sofa);
        VBO_Sofa_p.update(V_Sofa_p);
        VBO_Sofa_tex.update(V_Sofa_tex);
        glUniformMatrix4fv(program.uniform("size"), 1, GL_FALSE, size.data());
        glUniformMatrix4fv(program.uniform("rotation"), 1, GL_FALSE, rotation.data());
        glUniformMatrix4fv(program.uniform("transition"), 1, GL_FALSE, transition.data());
        glUniform1i(program.uniform("tex"), 2);
        program.bindVertexAttribArray("position",VBO_Sofa);
        program.bindVertexAttribArray("normal",VBO_Sofa_p);
        program.bindVertexAttribArray("texcoord",VBO_Sofa_tex);
        glUniform3f(program.uniform("triangleColor"), 0.5f, 0.5f, 0.5f);
        glDrawArrays(GL_TRIANGLES, 0, sofa_count);

        //Floor
        VBO_BG.update(floor_V);
        VBO_BG_p.update(floor_normal);
        VBO_BG_tex.update(floor_tex);
        glUniformMatrix4fv(program.uniform("size"), 1, GL_FALSE, size.data());
        glUniformMatrix4fv(program.uniform("rotation"), 1, GL_FALSE, rotation.data());
        glUniformMatrix4fv(program.uniform("transition"), 1, GL_FALSE, transition.data());
        glUniform1i(program.uniform("tex"), 4);
        program.bindVertexAttribArray("position",VBO_BG);
        program.bindVertexAttribArray("normal",VBO_BG_p);
        program.bindVertexAttribArray("texcoord",VBO_BG_tex);
        glUniform3f(program.uniform("triangleColor"), 1.0f, 1.0f, 1.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3600);


        //Table 
        VBO_Bumpy.update(V_Bumpy);
        VBO_Bumpy_p.update(V_Bumpy_p);
        VBO_Bumpy_f.update(V_Bumpy_f);
        glUniformMatrix4fv(program.uniform("size"), 1, GL_FALSE, size.data());
        glUniformMatrix4fv(program.uniform("rotation"), 1, GL_FALSE, rotation.data());
        glUniformMatrix4fv(program.uniform("transition"), 1, GL_FALSE, transition.data());
        glUniform1i(program.uniform("tex"), 3);
        program.bindVertexAttribArray("position",VBO_Bumpy);
        program.bindVertexAttribArray("normal",VBO_Bumpy_p);
        program.bindVertexAttribArray("texcoord",VBO_Bumpy_f);
        glUniform3f(program.uniform("triangleColor"), 0.5f, 0.5f, 0.5f);
        glDrawArrays(GL_TRIANGLES, 0, table_count);



        //Ball
        VBO_Ball.update(V_Ball);
        VBO_Ball_p.update(V_Ball_p);
        VBO_Ball_tex.update(V_Ball_tex);
        glUniformMatrix4fv(program.uniform("size"), 1, GL_FALSE, size.data());
        glUniformMatrix4fv(program.uniform("rotation"), 1, GL_FALSE, rotation.data());
        glUniformMatrix4fv(program.uniform("transition"), 1, GL_FALSE, transition.data());
        glUniform1i(program.uniform("tex"), 4);
        program.bindVertexAttribArray("position",VBO_Ball);
        program.bindVertexAttribArray("normal",VBO_Ball_p);
        program.bindVertexAttribArray("texcoord",VBO_Ball_tex);
        glUniform3f(program.uniform("triangleColor"), 1.0f, 1.0f, 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, ball_count);


        //Bunny
        VBO_Bunny.update(V_Bunny);
        VBO_Bunny_p.update(V_Bunny_p);
        VBO_Bunny_f.update(V_Bunny_f);
        glUniformMatrix4fv(program.uniform("size"), 1, GL_FALSE, size.data());
        glUniformMatrix4fv(program.uniform("rotation"), 1, GL_FALSE, rotation.data());
        glUniformMatrix4fv(program.uniform("transition"), 1, GL_FALSE, transition.data());
        glUniform1i(program.uniform("tex"), 4);
        program.bindVertexAttribArray("position",VBO_Bunny);
        program.bindVertexAttribArray("normal",VBO_Bunny_p);
        program.bindVertexAttribArray("texcoord",VBO_Bunny_f);
        glUniform3f(program.uniform("triangleColor"), 1.0f, 1.0f, 1.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3000);

        //Fire place
        VBO_Fire.update(V_Fire);
        VBO_Fire_p.update(V_Fire_p);
        VBO_Fire_tex.update(V_Fire_tex);
        glUniformMatrix4fv(program.uniform("size"), 1, GL_FALSE, size.data());
        glUniformMatrix4fv(program.uniform("rotation"), 1, GL_FALSE, rotation.data());
        glUniformMatrix4fv(program.uniform("transition"), 1, GL_FALSE, transition.data());
        program.bindVertexAttribArray("position",VBO_Fire);
        program.bindVertexAttribArray("normal",VBO_Fire_p);
        program.bindVertexAttribArray("texcoord",VBO_Fire_tex);
        
        //brick
        glUniform1i(program.uniform("tex"), 5);
        glUniform3f(program.uniform("triangleColor"), 1.0f, 0.5f, 0.5f);
        glDrawArrays(GL_TRIANGLES, 0, fire_count5);

        //molding
        glUniform1i(program.uniform("tex"), 3);
        glUniform3f(program.uniform("triangleColor"), 1.0f, 1.0f, 1.0f);
        glDrawArrays(GL_TRIANGLES, fire_count5, fire_count7-fire_count5);

        //brick
        glUniform1i(program.uniform("tex"), 5);
        glUniform3f(program.uniform("triangleColor"), 1.0f, 0.5f, 0.5f);
        glDrawArrays(GL_TRIANGLES, fire_count7, fire_count8-fire_count7);

        //concrete
        glUniform1i(program.uniform("tex"), 4);
        glUniform3f(program.uniform("triangleColor"), 1.0f, 1.0f, 1.0f);
        glDrawArrays(GL_TRIANGLES, fire_count8, fire_count9-fire_count8);

        //brick
        glUniform1i(program.uniform("tex"), 5);
        glUniform3f(program.uniform("triangleColor"), 1.0f, 0.5f, 0.5f);
        glDrawArrays(GL_TRIANGLES, fire_count9, fire_count11-fire_count9);

        //mantel
        glUniform1i(program.uniform("tex"), 3);
        glUniform3f(program.uniform("triangleColor"), 1.0f, 1.0f, 1.0f);
        glDrawArrays(GL_TRIANGLES, fire_count11, fire_count12 -fire_count11);

        //molding
        glUniform1i(program.uniform("tex"), 3);
        glUniform3f(program.uniform("triangleColor"), 1.0f, 1.0f, 1.0f);
        glDrawArrays(GL_TRIANGLES, fire_count12, fire_count13-fire_count12);



        // Draw cubes
        for (int i=0; i < Obj.size() ; i++){

            Matrix4f size = re_size(Obj[i].size);
            Matrix4f rotation = rotate(Obj[i].rx, Obj[i].ry, Obj[i].rz);
            Matrix4f transition = translate(Obj[i].tx, Obj[i].ty, Obj[i].tz);

            glUniformMatrix4fv(program.uniform("size"), 1, GL_FALSE, size.data());
            glUniformMatrix4fv(program.uniform("rotation"), 1, GL_FALSE, rotation.data());
            glUniformMatrix4fv(program.uniform("transition"), 1, GL_FALSE, transition.data());

            if (Obj[i].type == 1){
                for (int j=0; j < 36 ; j++){
                    V_Unit.col(j) << unitcube.col(j);
                    V_Unit_p.col(j) << Unit_Phong.col(j);
                    V_Unit_f.col(j) << cube_tex.col(j);
                }
            }
            else if (Obj[i].type == 2){
                for (int j=0; j < 36 ; j++){
                    V_Unit.col(j) << unitcube.col(j);
                    V_Unit_p.col(j) << Unit_Phong.col(j);
                    V_Unit_f.col(j) << cube_tex2.col(j);
                }
            }

            VBO_Unit.update(V_Unit);
            VBO_Unit_f.update(V_Unit_f);
            VBO_Unit_p.update(V_Unit_p);

            program.bindVertexAttribArray("position",VBO_Unit);
            program.bindVertexAttribArray("normal",VBO_Unit_p);
            program.bindVertexAttribArray("texcoord",VBO_Unit_f);

            //red
            if (Obj[i].color == 1){
                glUniform3f(program.uniform("triangleColor"), 1.0f, 0.0f, 0.0f);
                glUniform1i(program.uniform("tex"), 0);
                glDrawArrays(GL_TRIANGLES, 0, 36);
            }

            //black
            if (Obj[i].color == 2){
                glUniform3f(program.uniform("triangleColor"), 1.0f, 1.0f, 1.0f);
                glUniform1i(program.uniform("tex"), 7);
                glDrawArrays(GL_TRIANGLES, 0, 36);
            }
            //pink
            if (Obj[i].color == 3){
                glUniform3f(program.uniform("triangleColor"), 1.0f, 0.7f, 0.7f);
                glUniform1i(program.uniform("tex"), 6);
                glDrawArrays(GL_TRIANGLES, 0, 36);
            }
            //green
            if (Obj[i].color == 4){
                glUniform3f(program.uniform("triangleColor"), 0.0f, 1.0f, 0.0f);
                glUniform1i(program.uniform("tex"), 8);
                glDrawArrays(GL_TRIANGLES, 0, 36);
            }
            //blue
            if (Obj[i].color == 5){
                glUniform3f(program.uniform("triangleColor"), 0.0f, 0.0f, 1.0f);
                glUniform1i(program.uniform("tex"), 0);
                glDrawArrays(GL_TRIANGLES, 0, 36);
            }
            //orange
            if (Obj[i].color == 6){
                glUniform3f(program.uniform("triangleColor"), 1.0f, 0.6f, 0.0f);
                glUniform1i(program.uniform("tex"), 0);
                glDrawArrays(GL_TRIANGLES, 0, 36);
            }
            //white
            if (Obj[i].color == 7){
                glUniform3f(program.uniform("triangleColor"), 1.0f, 1.0f, 1.0f);
                glUniform1i(program.uniform("tex"), 0);
                glDrawArrays(GL_TRIANGLES, 0, 36);
            }
        }
        
        //Draw fireballs
        for (int i=0; i < Obj2.size() ; i++){

            Matrix4f size = re_size(Obj2[i].size);
            Matrix4f rotation = rotate(Obj2[i].rx, Obj2[i].ry, Obj2[i].rz);
            Matrix4f transition = translate(Obj2[i].tx, Obj2[i].ty, Obj2[i].tz);

            glUniformMatrix4fv(program.uniform("size"), 1, GL_FALSE, size.data());
            glUniformMatrix4fv(program.uniform("rotation"), 1, GL_FALSE, rotation.data());
            glUniformMatrix4fv(program.uniform("transition"), 1, GL_FALSE, transition.data());
            glUniform1i(program.uniform("tex"), 4);

            program.bindVertexAttribArray("position",VBO_Ball);
            program.bindVertexAttribArray("normal",VBO_Ball_p);
            program.bindVertexAttribArray("texcoord",VBO_Ball_tex);

            //init
            if (Obj2[i].color == 7){
                glUniform3f(program.uniform("triangleColor"), 1.0f, 1.0f, 1.0f);
                glDrawArrays(GL_TRIANGLES, 0, ball_count);
            }

            //red
            if (Obj2[i].color == 1){
                glUniform3f(program.uniform("triangleColor"), 1.0f, 0.0f, 0.0f);
                glDrawArrays(GL_TRIANGLES, 0, ball_count);
            }

            //Orange
            if (Obj2[i].color == 6){
                glUniform3f(program.uniform("triangleColor"), 1.0f, 0.5f, 0.0f);
                glDrawArrays(GL_TRIANGLES, 0, ball_count);
            }
            //Yellow
            if (Obj2[i].color == 2){
                glUniform3f(program.uniform("triangleColor"), 1.0f, 1.0f, 0.0f);
                glDrawArrays(GL_TRIANGLES, 0, ball_count);
            }
            //green
            if (Obj2[i].color == 5){
                glUniform3f(program.uniform("triangleColor"), 0.0f, 1.0f, 0.0f);
                glDrawArrays(GL_TRIANGLES, 0, ball_count);
            }
            //blue
            if (Obj2[i].color == 4){
                glUniform3f(program.uniform("triangleColor"), 0.0f, 0.0f, 1.0f);
                glDrawArrays(GL_TRIANGLES, 0, ball_count);
            }
            //indigo
            if (Obj2[i].color == 3){
                glUniform3f(program.uniform("triangleColor"), 0.2f, 0.0f, 0.7f);
                glDrawArrays(GL_TRIANGLES, 0, ball_count);
            }
            //violet
            if (Obj2[i].color == 0){
                glUniform3f(program.uniform("triangleColor"), 0.7f, 0.0f, 0.9f);
                glDrawArrays(GL_TRIANGLES, 0, ball_count);
            }
        }

        //event

        Matrix4f size_event = re_size(3);
        Matrix4f rotation_event = rotate(0.0f, 0.0f, 0.0f);

        Matrix3f cam_R1(3,3);
        cam_R1 <<
        cos(pi * cam_ud / 180), 0, sin(pi * cam_ud / 180),
        0, 1, 0,
        -sin(pi * cam_ud / 180), 0, cos(pi * cam_ud / 180);


        Matrix3f cam_R2(3,3);
        cam_R2 <<
        1, 0, 0,
        0, cos(pi * cam_rl / 180), -sin(pi * cam_rl / 180),
        0, sin(pi * cam_rl / 180), cos(pi * cam_rl / 180);

        Vector3f temp(0,0,1);
        temp = cam_R1*cam_R2*temp;
        Vector3f CameraDirection = temp.normalized();
        Vector3f front_loc = changed_eye - 5 * CameraDirection;

        Matrix4f transition_event = translate(front_loc(0), front_loc(1) - 2, front_loc(2));

        glUniformMatrix4fv(program.uniform("size"), 1, GL_FALSE, size_event.data());
        glUniformMatrix4fv(program.uniform("rotation"), 1, GL_FALSE, rotation_event.data());
        glUniformMatrix4fv(program.uniform("transition"), 1, GL_FALSE, transition_event.data());

        for (int k=0; k <2 ; k++){
            for (int j=0; j < 36 ; j++){
                Vector3f location_dif(k - 0.5, 0, 0);
                V_Unit.col(36*k + j) << unitcube.col(j) + location_dif;
                V_Unit_p.col(36*k + j) << Unit_Phong.col(j);
                V_Unit_f.col(36*k + j) << zero_tex.col(j);
            }
        }

        VBO_Unit.update(V_Unit);
        VBO_Unit_f.update(V_Unit_f);
        VBO_Unit_p.update(V_Unit_p);

        glUniform1i(program.uniform("tex"), 4);

        program.bindVertexAttribArray("position",VBO_Unit);
        program.bindVertexAttribArray("normal",VBO_Unit_p);
        program.bindVertexAttribArray("texcoord",VBO_Unit_f);

        if (select_i == 10){
            glUniform3f(program.uniform("triangleColor"), 1.0f, 0.0f, 0.0f);
            glDrawArrays(GL_TRIANGLES, 0, 36);

            glUniform3f(program.uniform("triangleColor"), 0.0f, 0.0f, 0.0f);
            glDrawArrays(GL_TRIANGLES, 36, 36);
        }
        if (select_i == 20){
            glUniform3f(program.uniform("triangleColor"), 0.0f, 0.0f, 0.0f);
            glDrawArrays(GL_TRIANGLES, 0, 36);

            glUniform3f(program.uniform("triangleColor"), 0.75f, 0.56f, 0.56f);
            glDrawArrays(GL_TRIANGLES, 36, 36);
            
        }
        if (select_i == 30){
            glUniform3f(program.uniform("triangleColor"), 0.75f, 0.56f, 0.56f);
            glDrawArrays(GL_TRIANGLES, 0, 36);

            glUniform3f(program.uniform("triangleColor"), 0.0f, 1.0f, 0.0f);
            glDrawArrays(GL_TRIANGLES, 36, 36);
            
        }
        if (select_i == 40){
            glUniform3f(program.uniform("triangleColor"), 0.0f, 1.0f, 0.0f);
            glDrawArrays(GL_TRIANGLES, 0, 36);

            glUniform3f(program.uniform("triangleColor"), 0.0f, 0.0f, 1.0f);
            glDrawArrays(GL_TRIANGLES, 36, 36);
            
        }
        if (select_i == 50){
            glUniform3f(program.uniform("triangleColor"), 0.0f, 0.0f, 1.0f);
            glDrawArrays(GL_TRIANGLES, 0, 36);

            glUniform3f(program.uniform("triangleColor"), 1.0f, 0.5f, 0.0f);
            glDrawArrays(GL_TRIANGLES, 36, 36);
            
        }
        if (select_i == 60){
            glUniform3f(program.uniform("triangleColor"), 1.0f, 0.5f, 0.0f);
            glDrawArrays(GL_TRIANGLES, 0, 36);

            glUniform3f(program.uniform("triangleColor"), 1.0f, 1.0f, 1.0f);
            glDrawArrays(GL_TRIANGLES, 36, 36);
            
        }

        bool ending = false;

        //Ending
        if (Obj2[0].color == 1 && Obj2[1].color == 6 && Obj2[2].color == 2 && Obj2[3].color == 5 && Obj2[4].color == 4 && Obj2[5].color == 3 && Obj2[6].color == 0){
            ending = true;
        }
        //Stair
        if (ending){
            VBO_Stair.update(V_Stair);
            VBO_Stair_p.update(V_Stair_p);
            VBO_Stair_tex.update(V_Stair_tex);
            glUniformMatrix4fv(program.uniform("size"), 1, GL_FALSE, size.data());
            glUniformMatrix4fv(program.uniform("rotation"), 1, GL_FALSE, rotation.data());
            glUniformMatrix4fv(program.uniform("transition"), 1, GL_FALSE, transition.data());
            glUniform1i(program.uniform("tex"), 4);
            program.bindVertexAttribArray("position",VBO_Stair);
            program.bindVertexAttribArray("normal",VBO_Stair_p);
            program.bindVertexAttribArray("texcoord",VBO_Stair_tex);
            glUniform3f(program.uniform("triangleColor"), 0.5f, 0.5f, 0.5f);
            glDrawArrays(GL_TRIANGLES, 0, stair_count);
        }

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    // Deallocate opengl memory
    program.free();
    VAO.free();
    VBO_Bumpy.free();
    VBO_Bumpy_f.free();
    VBO_Bumpy_p.free();
    VBO_Bunny.free();
    VBO_Bunny_f.free();
    VBO_Bunny_p.free();
    VBO_Unit.free();
    VBO_Unit_f.free();
    VBO_Unit_p.free();
    VBO_Clock.free();
    VBO_Clock_p.free();
    VBO_Clock_tex.free();
    VBO_Sofa.free();
    VBO_Sofa_p.free();
    VBO_Sofa_tex.free();
    VBO_BG.free();
    VBO_BG_p.free();
    VBO_BG_tex.free();
    VBO_Fire.free();
    VBO_Fire_p.free();
    VBO_Fire_tex.free();
    VBO_Ball.free();
    VBO_Ball_p.free();
    VBO_Ball_tex.free();
    VBO_Stair.free();
    VBO_Stair_p.free();
    VBO_Stair_tex.free();

    // Deallocate glfw internals
    glfwTerminate();
    return 0;
}
