#include <stdio.h>
#include <string>

#include <easy3d/viewer/viewer.h>

int main()
{
   printf("int 存储大小 : %lu \n", sizeof(int));

   std::string s = "1121";

   std::cout << s;
   

   // easy3d::Viewer viewer("abc");

   // if(!viewer.add_model("G:\\libs\\Easy3D\\resources\\data/bunny.ply")){
   //    printf("int 存储大小 : %lu \n", sizeof(int));
   // }

   // viewer.run();

   
   return 0;
}