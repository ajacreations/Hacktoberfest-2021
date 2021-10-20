#include<stdio.h>
int main()
{
	int r,c,p,q,i,j;
	printf("enter row and column for first matrix");
	scanf("%d%d",&r,&c);
	int a[10][10],b[10][10];
	
	printf("enter number for first matrix");
		for(i=0;i<r;i++)
		{
			for(j=0;j<c;j++)
			{
			//printf("enter number for first matrix");
			scanf("%d",&a[i][j]);
		}}
		printf("origional matrix\n");
			for(i=0;i<r;i++)
		{
			for(j=0;j<c;j++)
			{
				b[j][i]=a[i][j];
			printf("%d ",a[i][j]);}
			printf("\n");
		}
		printf("transpose matrix\n");
			for(i=0;i<c;i++)
		{
			for(j=0;j<r;j++)
			printf("%d ",b[i][j]);
			printf("\n");
		}
	}

	
