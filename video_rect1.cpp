/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - video_rect1.cpp
// TOPIC: video handling for motion
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include<unistd.h>
using namespace std;
using namespace cv;

//function declaration
Mat normalisation (Mat imageG);
void  LKTracker(Mat,Mat,Mat, Mat);


static int x=300,y=200;// set the position of the ball;
int main( int argc, const char** argv )
{
			VideoCapture cap(0); // open the default camera

			    if(!cap.isOpened())  // check if we succeeded
			        return -1;
				Mat frame,frame1, gray, gray1;
				Size size(500, 400);
				//namedWindow("video", 1);
				int k = 0;
			for(;;)
			{
		
			cap >> frame;//capture the first frame 
			resize(frame,frame,size);
			flip(frame,frame,1);//fliping the frame for realtime movement and not inverse image 
	rectangle(frame, Point(250,200), Point(0,0), Scalar( 255, 255, 0 ), 5);//draw the rectangle for the controller area				
	
			cvtColor(frame, gray, CV_BGR2GRAY);
			circle(frame,Point(x,y),50,Scalar(51,153,255),-1);//drawing the circle
					
			GaussianBlur(gray,gray, Size(35,35),0, 0, BORDER_DEFAULT );
			 		
		
		 if(k % 4==0)//capturing the forth frame
		{
			cap >> frame1;
			resize(frame1,frame1,size);
			flip(frame1,frame1,1);
			cvtColor(frame1, gray1, CV_BGR2GRAY);
			
			//GaussianBlur(gray1,gray1, Size(5,5),0, 0, BORDER_DEFAULT );
		
		}
		if(!frame.data)
		{
			printf("Error: no frame data.\n");
			break;
		}


		Mat Ix(gray.rows-1, gray.cols-1, CV_32FC1, Scalar::all(0));
		Mat Iy(gray.rows-1, gray.cols-1, CV_32FC1, Scalar::all(0));
		Mat It(gray.rows-1, gray.cols-1, CV_32FC1, Scalar::all(0));
		Mat Ixn(gray.rows-1, gray.cols-1, CV_8UC1, Scalar::all(0));
		Mat Iyn(gray.rows-1, gray.cols-1, CV_8UC1, Scalar::all(0));
		Mat Itn(gray.rows-1, gray.cols-1, CV_8UC1, Scalar::all(0));

		k++;
	
		if (k > 1)
		{
			for ( int i = 0; i <200; i++)
			{
			for ( int j = 0; j <250; j++)
			{
					Ix.at<float>(i,j) = (float)(gray.at<uchar>(i,j + 1) - gray.at<uchar>(i,j) +
							gray.at<uchar>(i + 1, j + 1) - gray.at<uchar>(i + 1,j) + 
							gray1.at<uchar>(i,j + 1) - gray1.at<uchar>(i,j) +
							gray1.at<uchar>(i + 1, j + 1) - gray1.at<uchar>(i + 1,j) ) / 4.0;
					Iy.at<float>(i,j) = (float)(gray.at<uchar>(i + 1,j) - gray.at<uchar>(i,j) +
							gray.at<uchar>(i + 1, j + 1) - gray.at<uchar>(i,j + 1) + 
							gray1.at<uchar>(i + 1,j) - gray1.at<uchar>(i,j) +
							gray1.at<uchar>(i + 1, j + 1) - gray1.at<uchar>(i,j + 1) ) / 4.0;
					It.at<float>(i,j) = (float)(gray1.at<uchar>(i,j) - gray.at<uchar>(i,j) +
							gray1.at<uchar>(i, j + 1) - gray.at<uchar>(i,j + 1) +
							gray1.at<uchar>(i + 1,j) - gray.at<uchar>(i + 1,j) +
							gray1.at<uchar>(i + 1, j + 1) - gray.at<uchar>(i + 1,j + 1))  / 4.0;
					//It.at<float>(i,j) = (float)(gray.at<uchar>(i,j)-gray1.at<uchar>(i,j));				

				}
			}
			Ixn=normalisation(Ix);
			Iyn=normalisation(Iy);
			Itn=normalisation(It);
		}
		
	

		LKTracker(Ix,Iy,It, frame);//calling LK TRACKER 

	
		imshow("video", frame);	
		imshow("Ixn", Ixn);	
		imshow("Iyn", Iyn);	
		imshow("Itn", Itn);
		if(waitKey(10) >= 0) break;	

	}//end of infinite loop
	

	
	
}



Mat normalisation (Mat imageG)
{
	double min, max;
	minMaxLoc(imageG, &min, &max);
	Mat normimage (imageG.rows, imageG.cols, CV_8UC1, Scalar::all(0));
	Mat abs_normimage(imageG.rows, imageG.cols, CV_8UC1, Scalar::all(0));
	float koeff = (float)( 255.0/(max - min));


	for ( int i = 0; i <200; i++)
	{
	for ( int j = 0; j <250; j++)
	{			
	normimage.at<uchar>(i, j) = (uchar)((imageG.at<float>(i,j)-min) * koeff);
	}
	}
	return normimage;
}




void LKTracker(Mat Ix1,Mat Iy1,Mat It1, Mat frame)
{
		Mat A(2,2, CV_32FC1, Scalar::all(0));
		Mat inv_A(2,2, CV_64FC1, Scalar::all(0));
		Mat b(2,1, CV_32FC1, Scalar::all(0));
		Mat v(2,1, CV_32FC1, Scalar::all(0));
		int magx=0,magy=0;
			for ( int i = 0; i < 200; i=i+10 )
			{
				for ( int j = 0; j < 250; j=j+10 )
				{
					int k=0,l=0;
					//Calculating the A , b and V					
						A.at<float>(0,0) = A.at<float>(0,0) + (Ix1.at<float>(i + k,j + l) * Ix1.at<float>(i + k,j + l));
						A.at<float>(0,1) = A.at<float>(0,1) + (Ix1.at<float>(i + k,j + l) * Iy1.at<float>(i + k,j + l));
						A.at<float>(1,0) = A.at<float>(1,0) + (Ix1.at<float>(i + k,j + l) * Iy1.at<float>(i + k,j + l));
						A.at<float>(1,1) = A.at<float>(1,1) + (Iy1.at<float>(i + k,j + l) * Iy1.at<float>(i + k,j + l));

						b.at<float>(0,0) = b.at<float>(0,0) - (It1.at<float>(i + k,j + l) * Ix1.at<float>(i + k,j + l));
						b.at<float>(1,0) = b.at<float>(1,0) - (It1.at<float>(i + k,j + l) * Iy1.at<float>(i + k,j + l));	
						}//end of inner loop
					}//end of outer loop 	
					invert (A, inv_A, DECOMP_LU);
					//inv_A=A.inv(DECOMP_LU);	
					v = inv_A*(b);//simple matrix mul
					//cout << "V_x = "<< v.at<float>(0,0) << endl << "V_y = "<< v.at<float>(1,0) << endl;
					//cout << "It1 = "<< It1.at<float>(i,j)  << endl;
			

			for ( int i = 0; i < 200; i=i+10 )
			{
				for ( int j = 0; j < 250; j=j+10 )
				{
				//Randomised Color for arrows
          			     int b1 = rand() % 255 + 0;
			         	     int g = rand() % 255 + 0;
	     			     int r = rand() % 255 + 0;

		
						//Drawing the reference frame 
						line(frame, Point(250,200), Point (250 + 250, 200), Scalar(0,255,0),1,8, 0);//x-right
						line(frame, Point(250,200), Point (250 - 250, 200), Scalar(0,255,0),1,8, 0);//x-left
						line(frame, Point(250,200), Point (250, 200 + 200), Scalar(0,255,0),1,8, 0);//y-up
						line(frame, Point(250,200), Point (250, 200 - 200), Scalar(0,255,0),1,8, 0);//y-down



				if(It1.at<float>(i,j) >60.0)	
				{
		//drawing the vectors
		//line(frame,Point(125,100), Point ((125 + ((v.at<float>(0,0))*17)), (100 + ((v.at<float>(1,0))*17))),  Scalar(0,0,255),1,8, 0);


		//Drawing the Circle for arrow reference 
		circle(frame,Point(j,i),5,Scalar(b1,g,r),-1);
						
	      

		





			
 			//x1=x1-v.at<float>(0,0);
 		         //y1=y1-v.at<float>(1,0);




int vxlog = (int) log(10*abs(v.at<float>(0,0)));
int vylog =  (int)log(10*abs(v.at<float>(1,0)));

//To check the position of the ball
cout << "x = " << vxlog << "y =" << vylog<< endl;			


//Checking the boundary conditions of the ball
		if(x>=50&&x<=450&&y>=50&&y<=350)
		{

			if(v.at<float>(0,0) < 0.0)			
			{ 
				x = x+vxlog;
			}
				else
				{
					x = x-vxlog;
				}
					if(v.at<float>(1,0) < 0.0)
					{
				 		y = y+vylog;
					}
						else
						{
							y = y-vxlog;
						}

				//x = x - v.at<float>(0,0);	
				//y = y - v.at<float>(1,0);
		}

		else
		{
			if(x<50)
			x=60;
			if(y<50)
			y=60;
			if(x>450)
			x=420;
			if(y>350)
			y=320;
		
		}				
	} //end of if


		else if((It1.at<float>(i,j) <60.0))
		{
		putText(frame,"MOVE YOUR HAND ", Point(10,80),FONT_HERSHEY_SIMPLEX,1, Scalar(255,0,0),2,8,false);
		putText(frame,"IN THIS REGION ", Point(10,120),FONT_HERSHEY_SIMPLEX,1, Scalar(255,0,0),2,8,false);
		}
		

 		
}//end of inner loop
} //end of for outer loop

} //end of function LK TRACKER
