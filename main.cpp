#pragma comment (lib, "libgsl.a")
/* From GSL */

/* From opencv*/
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>

#include<iostream>
#include<stdlib.h>
#include<stdio.h>
#include<time.h>
using namespace std;
using namespace cv;
int get_min_index(double *array, int length, double _value);
/*********************�ṹ��************************/
// ���ӽṹ��
typedef struct particle {
	double x;			// ��ǰx����
	double y;			// ��ǰy����
	double scale;		// ���ڱ���ϵ��
	double xPre;			// x����Ԥ��λ��
	double yPre;			// y����Ԥ��λ��
	double scalePre;		// ����Ԥ�����ϵ��
	double xOri;			// ԭʼx����
	double yOri;			// ԭʼy����
	int width;			// ԭʼ�������
	int height;			// ԭʼ����߶�
	//Rect rect;			// ԭʼ�����С
	MatND hist;			// �������������ֱ��ͼ
	double weight;		// �����ӵ�Ȩ��
} PARTICLE;

/*************************ȫ�ֱ���*************************/
char* video_file_name = "soccer.avi";
#define NUM_PARTICLE 50 //������
/************************���ص�����*************************/
Rect roiRect;//ѡȡ������
Point startPoint;//���
Point endPoint;//�յ�
Mat current_frame;
Mat roiImage;
Mat hsv_roiImage;
bool downFlag = false;//���±�־λ
bool upFlag = false;//�����־λ
bool getTargetFlag = false;
//rect����

Mat regionExtraction(int xRoi, int yRoi,
	int widthRoi, int heightRoi)
{
	//������ԭͼ��ͬ��С��Mat
	Mat roiImage;// (srcImage.rows, srcImage.cols, CV_8UC3);
	//��ȡ����Ȥ����
	roiImage = current_frame(Rect(xRoi, yRoi, widthRoi, heightRoi)).clone();
	imshow("set-roi", roiImage);
	return roiImage;
}

//����¼��ص�����
void MouseEvent(int event, int x, int y, int flags, void* win_name)
{
	//������£�ȡ��ǰλ��
	if (event == EVENT_LBUTTONDOWN){
		downFlag = true;
		getTargetFlag = false;
		startPoint.x = x;
		startPoint.y = y;
	}
	//����ȡ��ǰλ����Ϊ�յ�
	if (event == EVENT_LBUTTONUP) {
		upFlag = true;
		endPoint.x = x;
		endPoint.y = y;
		//�յ���ֵ�޶�
		if (endPoint.x > current_frame.cols)endPoint.x = current_frame.cols;
		if (endPoint.y > current_frame.cols)endPoint.y = current_frame.rows;
	}
	//��ʾ����
	if (downFlag == true && upFlag == false){
		Point tempPoint;
		tempPoint.x = x;
		tempPoint.y = y;
		Mat tempImage = current_frame.clone();//ȡԭͼ����
		//�þ��α��
		rectangle(tempImage, startPoint, tempPoint, Scalar(0, 0, 255), 2, 3, 0);
		//imshow((char*)data, tempImage);
		imshow((char*)win_name, tempImage);
	}
	//����ѡȡ�겢�����
	if (downFlag == true && upFlag == true){
		//�����յ㲻��ͬʱ������ȡ����
		if (startPoint.x != endPoint.x&&startPoint.y != endPoint.y){
			startPoint.x = min(startPoint.x, endPoint.x);
			startPoint.y = min(startPoint.y, endPoint.y);
			roiRect = Rect(startPoint.x, startPoint.y, endPoint.x - startPoint.x, endPoint.y - startPoint.y);
			roiImage=regionExtraction(startPoint.x, startPoint.y,
				abs(startPoint.x - endPoint.x),
				abs(startPoint.y - endPoint.y));
		}
		downFlag = false;
		upFlag = false;
		getTargetFlag = true;
	}
}
/*************************���ӳ�ʼ��******************************************/
void particle_init(particle* particles,int _num_particle,MatND hist)
{
	for (int i = 0; i<_num_particle; i++)
	{
		//�������ӳ�ʼ�������е�Ŀ������
		particles[i].x = roiRect.x + 0.5 * roiRect.width;
		particles[i].y = roiRect.y + 0.5 * roiRect.height;
		particles[i].xPre = particles[i].x;
		particles[i].yPre = particles[i].y;
		particles[i].xOri = particles[i].x;
		particles[i].yOri = particles[i].y;
		//pParticles->rect = roiRect;
		particles[i].width = roiRect.width;
		particles[i].height = roiRect.height;
		particles[i].scale = 1.0;
		particles[i].scalePre = 1.0;
		particles[i].hist = hist;
		//Ȩ��ȫ��Ϊ0��
		particles[i].weight = 0;
	}
}
/************************����״̬ת�ƣ���λ������Ԥ�⣩***********************/
//��ض���
/* standard deviations for gaussian sampling in transition model */
#define TRANS_X_STD 1.0
#define TRANS_Y_STD 0.5
#define TRANS_S_STD 0.001
/* autoregressive dynamics parameters for transition model */
#define A1  2.0//2.0
#define A2  -1.0//-1.0
#define B0  1.0000
particle transition(particle p, int w, int h, gsl_rng* rng)
{
	//double rng_nu_x = rng.uniform(0., 1.);
	//double rng_nu_y = rng.uniform(0., 0.5);
	float x, y, s;
	particle pn;

	/* sample new state using second-order autoregressive dynamics */
	x = A1 * (p.x - p.xOri) + A2 * (p.xPre - p.xOri) +
		B0 * gsl_ran_gaussian(rng, TRANS_X_STD)/*rng.gaussian(TRANS_X_STD)*/ + p.xOri;  //�����������һʱ�̵�x
	pn.x = MAX(0.0, MIN((float)w - 1.0, x));
	y = A1 * (p.y - p.yOri) + A2 * (p.yPre - p.yOri) +
		B0 * gsl_ran_gaussian(rng, TRANS_Y_STD)/*rng.gaussian(TRANS_Y_STD)*/ + p.yOri;
	pn.y = MAX(0.0, MIN((float)h - 1.0, y));
	s = A1 * (p.scale - 1.0) + A2 * (p.scalePre - 1.0) +
		B0 * gsl_ran_gaussian(rng, TRANS_S_STD)/*rng.gaussian(TRANS_S_STD)*/ + 1.0;
	pn.scale = MAX(0.1, s);
	pn.xPre = p.x;
	pn.yPre = p.y;
	pn.scalePre = p.scale;
	pn.xOri = p.xOri;
	pn.yOri = p.yOri;
	pn.width = p.width;
	pn.height = p.height;
	//pn.hist = p.hist;
	pn.weight = 0;

	return pn;
}
/*************************����Ȩ�ع�һ��****************************/
void normalize_weights(particle* particles, int n)
{
	float sum = 0;
	int i;

	for (i = 0; i < n; i++)
		sum += particles[i].weight;
	for (i = 0; i < n; i++)
		particles[i].weight /= sum;
}
/***********************ֱ��ͼ����********************************/
// ֱ��ͼ
int hbins = 10, sbins = 10, vbin = 20;  //180 256 10
int histSize[] = { hbins, sbins };//vbin
//h�ķ�Χ
float hranges[] = { 0, 180 };
//s�ķ�Χ
float sranges[] = { 0, 256 };
float vranges[] = { 0, 256 };
//һ��ֻ�Ƚ�hsv��h��s����ͨ���͹���
const float* ranges[] = { hranges, sranges }; 
// we compute the histogram from the 0-th and 1-st channels
int channels[] = { 0, 1 };

/*************************����Ȩ������******************************/
int particle_cmp(const void* p1,const void* p2)
{
	//����������qsort����������������ֵ: (1) <0ʱ��p1����p2ǰ��   (2)  >0ʱ��p1����p2����
	particle* _p1 = (particle*)p1;
	particle* _p2 = (particle*)p2;
	//������ɴ�С������
	return _p2->weight - _p1->weight;
}
/*************************�����ز���********************************/
void resample(particle* particles,particle* new_particles,int num_particles)
{
	//����ÿ�����ӵĸ����ۼƺ�
	double sum[NUM_PARTICLE], temp_sum = 0;
	int k = 0;
	for (int j = num_particles - 1; j >= 0; j--){
		temp_sum += particles[j].weight;
		sum[j] = temp_sum;
	}
	//Ϊÿ����������һ�����ȷֲ���0��1���������
	RNG sum_rng(time(NULL));
	double Ran[NUM_PARTICLE];
	for (int j = 0; j < num_particles; j++){
		sum_rng = sum_rng.next();
		Ran[j] = sum_rng.uniform(0., 1.);
	}
	//�����Ӹ����ۻ����������ҵ���С�Ĵ��ڸ�������������������Ƹ�����������һ�ε��µ����������� ����Ȩ�ظߵ����ӿ�ʼ��
	for (int j = 0; j <num_particles; j++){
		int copy_index = get_min_index(sum, num_particles, Ran[j]);
		new_particles[k++] = particles[copy_index];
		if (k == num_particles)
			break;
	}
	//�������Ĳ�����ɣ����������������������ԭ������������������Ȩ����ߵ����ӣ�ֱ�����������
	while (k < num_particles)
	{
		new_particles[k++] = particles[0]; //����Ȩֵ��ߵ�����
	}
	//�����������鸲�Ǿõ���������
	for (int i = 0; i<num_particles; i++)
	{
		particles[i] = new_particles[i];  //���������ӵ�particles
	}
}

/*****************************************************************/
int main()
{
	//cv::RNG rng; //�°�OPENCV�Դ������������
	Mat frame, hsv_frame;
	Vector<Mat> frames;
	//Ŀ���ֱ��ͼ
	MatND hist;
	VideoCapture capture(video_file_name);	// ��Ƶ�ļ�video_file_name

	int num_particles = NUM_PARTICLE; //������
	PARTICLE particles[NUM_PARTICLE];
	PARTICLE new_particles[NUM_PARTICLE];
	PARTICLE * pParticles;
	pParticles=particles;
	//particles = (particle*)malloc(num_particles * sizeof(particle));
	//PARTICLE *new_particles;
	//�����������
	gsl_rng* rng;
	gsl_rng_env_setup();
	rng = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(rng, time(NULL));
	//cv::RNG rng(time(NULL));
	float s;
	int i, j, k, w, h, x, y;
	//�ж���Ƶ�Ƿ��
	if (!capture.isOpened())
	{
		cout << "some thing wrong" << endl;
		system("pause");
		return -1;
	}
	//��ȡһ֡
	while (1){
		capture >> frame;
		if (frame.empty()){
			cout << "finish" << endl;
			break;
			//return -1;
		}
		//��������
		namedWindow("frame", CV_WINDOW_NORMAL);
		//����һ��ԭʼ֡������Ŀ��ص�������
		current_frame = frame.clone();
		setMouseCallback("frame", MouseEvent,"frame");
		frames.push_back(frame.clone());
		imshow("frame", frame);
		cvWaitKey(40);

		if (getTargetFlag == true){
			//Ŀ������ת����hsv�ռ�
			//cvtColor(frame, hsv_frame, COLOR_BGR2HSV);
			cvtColor(roiImage, hsv_roiImage, COLOR_BGR2HSV);
			//����Ŀ�������ֱ��ͼ
			calcHist(&hsv_roiImage, 1, channels, Mat(), hist, 2, histSize, ranges);
			normalize(hist, hist,0,1,NORM_MINMAX,-1,Mat());	// ��һ��L2
			//���ӳ�ʼ��
			particle_init(particles, num_particles, hist);
			//pParticles = particles;
			//for (int i = 0; i<num_particles; i++)
			//{
			//	//�������ӳ�ʼ�������е�Ŀ������
			//	pParticles->x = roiRect.x + 0.5 * roiRect.width;
			//	pParticles->y = roiRect.y + 0.5 * roiRect.height;
			//	pParticles->xPre = pParticles->x;
			//	pParticles->yPre = pParticles->y;
			//	pParticles->xOri = pParticles->x;
			//	pParticles->yOri = pParticles->y;
			//	//pParticles->rect = roiRect;
			//	pParticles->width = roiRect.width;
			//	pParticles->height = roiRect.height;
			//	pParticles->scale = 1.0;
			//	pParticles->scalePre = 1.0;
			//	pParticles->hist = hist;
			//	//Ȩ��ȫ��Ϊ0��
			//	pParticles->weight = 0;
			//	pParticles++;
			//}
		}
		else{
			continue;
		}
		while (1){
			//��ʼ����ɲ��ܽ�������
			capture >> frame;
			if (frame.empty()){
				cout << "finish" << endl;
				break;
				//return -1;
			}
			current_frame = frame.clone();
			frames.push_back(frame.clone());

			//cv::RNG rng;
			//double rng_num = rng.uniform(0., 1.);
			
			//��ÿ�����ӵĲ�����
			for (j = 0; j < num_particles; j++){
				//rng = rng.next();
				//�������ø�˹�ֲ��������������ÿ��������һ�ε�λ���Լ���Χ
				particles[j] = transition(particles[j], frame.cols, frame.rows, rng);
				s = particles[j].scale;
				
				//���������ɵ�������Ϣ��ȡ��Ӧframe�ϵ�����
				Rect imgParticleRect = Rect(std::max(0, std::min(cvRound(particles[j].x - 0.5*particles[j].width), cvRound(frame.cols - particles[j].width*s))),
					std::max(0, std::min(cvRound(particles[j].y - 0.5*particles[j].height), cvRound(frame.rows - particles[j].height*s))),
					cvRound(particles[j].width*s),
					cvRound(particles[j].height*s));

				Mat imgParticle = current_frame(imgParticleRect).clone();
				//��������ת����hsv�ռ�
				cvtColor(imgParticle, imgParticle, CV_BGR2HSV);
				//���������ֱ��ͼ
				calcHist(&imgParticle, 1, channels, Mat(), particles[j].hist, 2, histSize, ranges);
				//ֱ��ͼ��һ������0��1��
				normalize(particles[j].hist, particles[j].hist, 0, 1, NORM_MINMAX, -1, Mat());	// ��һ��L2
				//������ɫ�����ӿ�
				rectangle(frame, imgParticleRect, Scalar(255, 0, 0), 1, 8);
				imshow("particle", imgParticle);
				//�Ƚ�Ŀ���ֱ��ͼ���������������ֱ��ͼ,����particleȨ��
				particles[j].weight = exp(-100 * (compareHist(hist, particles[j].hist, CV_COMP_BHATTACHARYYA))); //CV_COMP_CORREL
				
				int jj = 0;
			}
			//��һ��Ȩ�� 
			normalize_weights(particles, num_particles);

			//�ز���
			//new_particles = resample(particles, num_particles);
			int np, k = 0;
			//�����Ӱ�Ȩ�شӸߵ�������
			qsort(particles, num_particles, sizeof(particle), &particle_cmp);
			//�ز���
			resample(particles, new_particles, num_particles);
			//double sum[NUM_PARTICLE],temp_sum=0;
			//for (int j = num_particles - 1; j >= 0; j--){
			//	temp_sum += particles[j].weight;
			//	sum[j] = temp_sum;
			//}
			//RNG sum_rng(time(NULL));
			//double Ran[NUM_PARTICLE];
			//for (int j = 0; j < num_particles; j++){
			//	sum_rng=sum_rng.next();
			//	Ran[j]=sum_rng.uniform(0., 1.);
			//}
			//for (int j = 0; j <num_particles; j++){
			//	int copy_index = get_min_index(sum, num_particles, Ran[j]);
			//	new_particles[k++] = particles[copy_index];
			//	if (k == num_particles)
			//		break;
			//}
			//while (k < num_particles)
			//{
			//	new_particles[k++] = particles[0]; //����Ȩֵ��ߵ�����
			//}
			//for (int i = 0; i<num_particles; i++)
			//{
			//	particles[i] = new_particles[i];  //���������ӵ�particles
			//}


		//  //������ز����㷨ò�Ʋ��Ǳ�׼�ģ����һ���goto���д�����
		//	for (int i = 0; i<num_particles; i++)
		//	{
		//		np = cvRound(particles[i].weight*1.0 * num_particles);
		//		for (int j = 0; j<np; j++)
		//		{
		//			new_particles[k++] = particles[i];
		//			if (k == num_particles)
		//				goto EXITOUT;
		//		}
		//	}
		//	while (k < num_particles)
		//	{
		//		new_particles[k++] = particles[0]; //����Ȩֵ��ߵ�����
		//	}
		//EXITOUT:
		//	for (int i = 0; i<num_particles; i++)
		//	{
		//		particles[i] = new_particles[i];  //���������ӵ�particles
		//	}

			//������
			qsort(particles, num_particles, sizeof(particle), &particle_cmp);
			// step 8: �������ӵ���������Ϊ���ٽ��
			//Rect_<double> rectTrackingTemp(0.0, 0.0, 0.0, 0.0);
			//rectTrackingTemp.x = cvRound(particles[0].x-0.5*particles[0].scale*particles[0].width);
			//rectTrackingTemp.y = cvRound(particles[0].y - 0.5*particles[0].scale*particles[0].height);
			//rectTrackingTemp.width = cvRound(particles[0].width*particles[0].scale);
			//rectTrackingTemp.height = cvRound(particles[0].height*particles[0].scale);
			
			//����ֱ��ȡȨ����ߵ���ΪĿ���ˣ���׼����Ӧ���ǰ���Ȩƽ��������Ŀ��λ��
			s = particles[0].scale;
			Rect rectTrackingTemp = Rect(std::max(0, std::min(cvRound(particles[0].x - 0.5*particles[0].width), cvRound(frame.cols - particles[0].width*s))),
				std::max(0, std::min(cvRound(particles[0].y - 0.5*particles[0].height), cvRound(frame.rows - particles[0].height*s))),
				cvRound(particles[0].width*s),
				cvRound(particles[0].height*s));
			//for (int i = 0; i < num_particles; i++){
			//	rectTrackingTemp.x += particles[i]
			//}
			rectangle(frame, rectTrackingTemp, Scalar(0, 0, 255), 1, 8, 0);
			imshow("frame", frame);
			cvWaitKey(40);
		}
	}
}

/*���ַ��������д��ڸ���ֵ����Сֵ����*/
int get_min_index(double *array, int length, double _value)
{
	int _index = (length - 1) / 2;
	int last_index = length - 1;
	int _index_up_limit = length - 1;
	int _index_down_limit = 0;
	//���жϼ�ֵ
	if (array[0] <= _value){
		return 0;
	}
	if (array[length - 1] > _value){
		return length - 1;
	}
	for (; _index != last_index;){
		//cout << _index << endl;
		last_index = _index;
		if (array[_index] > _value){
			_index = (_index_up_limit + _index) / 2;
			_index_down_limit = last_index;
		}
		else if (array[_index] < _value){
			_index = (_index_down_limit + _index) / 2;
			_index_up_limit = last_index;
		}
		else if (array[_index] == _value){
			_index--;
			break;
		}
	}
	//cout << "final result:" << endl;
	//cout << _index << endl;
	return _index;
}