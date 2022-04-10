/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package aitumourclasification;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

/**
 *
 * @author mortimer
 */
public class NeuralNetwork {
    final int netStructureLength;
    final int netStructureLengthMinusOne;
    final int allLength;
    final int[]totNeuron;//
    final int[]currentAll;//
    Random rnd = new Random();
    final int[]indexForAllWeightAndError;//
    final int[]indexForActivError;//
    final int activIn;
    final int[]netStructure;
    //int indexFirstLayerActiv;
    double[]allWeightsAndBiases;// in the form of bias, all weights leading into bias going down in array is oind down/left
    final int totalNeurons;
    final int totalNeuronsMinusLastLayer;
    double[] preSigActivs;
    //double histOfSquare[];
    double []postSigActivs;
    final String FILE_NAME = "networkInfo.txt";
    double timeStep;
    double previousMHat[];
    double previousVHat[];
    public  NeuralNetwork(int[]netStructure){
        this.netStructure = netStructure;
        this.netStructureLength=this.netStructure.length;
        this.netStructureLengthMinusOne=this.netStructureLength-1;
        int totalWeightsAndBiases=0;
        int totalNeurons=0;
        for(int i =0;i<netStructure.length;i++){
            totalNeurons=totalNeurons+this.netStructure[i];
            if(i==0){
                
            }else{
                totalWeightsAndBiases = totalWeightsAndBiases+this.netStructure[i]+(this.netStructure[i]*this.netStructure[i-1]);
            }
        }
        this.totalNeurons = totalNeurons;
        this.totalNeuronsMinusLastLayer=this.totalNeurons-this.netStructure[this.netStructureLengthMinusOne];
        this.allWeightsAndBiases=new double[totalWeightsAndBiases];
        this.allLength=totalWeightsAndBiases;
        this.activIn=this.totalNeurons-this.netStructure[this.netStructure.length-1];
        int currentIndex=0;
        for(int layerIndex=0;layerIndex<this.netStructure.length;layerIndex++){
            if(layerIndex!=0){
                for(int currentLayerIndex=0;currentLayerIndex<this.netStructure[layerIndex];currentLayerIndex++){
                    this.allWeightsAndBiases[currentIndex]=0.0;//Math.random()*0.001;//getRandDoubleBetweenOneAndMinusOne(this.netStructure[layerIndex-1]);//bias in current layer
                    //this.allWeightsAndBiases[currentIndex]=0;//DELETE
                    currentIndex++;
                    for(int prevLayerIndex=0;prevLayerIndex<this.netStructure[layerIndex-1];prevLayerIndex++){//weigths leading into bias
                        this.allWeightsAndBiases[currentIndex]=getRandDoubleBetweenOneAndMinusOne(this.netStructure[layerIndex-1],this.netStructure[layerIndex]);
                        currentIndex++;
                    }
                }
            }
        }
        this.feedThroughNet(new double[this.netStructure[0]]);
        double[]desiredOutput=new double[this.netStructure[this.netStructureLengthMinusOne]];
                //using 1/2 (act-des)^2
        double[]errorInLayer= new double[1];
        double[]errorInLayerAbove=new double[1];
        double sumOfErrorAndWeights;
        double grad[]=new double[this.allWeightsAndBiases.length];
        int indexForAllWhenDoingWeightsAndError;
        int indexWeightErrorChange;
        int indexForActivsError;
        int totalAllExplored;
        int indexForAllGrad;
        int totNeuronsExplored;
        int indexForActivsGrad;
        int activIndex;
        int[] arrayIndexForAllWhenDoingWeightsAndError=new int[this.netStructure.length];
        int[]arrayForIndexForActivsError= new int[this.netStructure.length];
        int[]arrayForIndexForAllGrad=new int[this.netStructure.length];
        int []arrayForTotNeuronsExplored=new int[this.netStructure.length];
        for(int layerIndex=this.netStructure.length-1;layerIndex>0;layerIndex--){
            errorInLayer=new double[this.netStructure[layerIndex]];
            if(layerIndex==this.netStructure.length-1){
                activIndex=this.totalNeurons-this.netStructure[this.netStructure.length-1];
                for(int neuronInLayer=0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                    errorInLayer[neuronInLayer]=(this.postSigActivs[activIndex]-desiredOutput[neuronInLayer])*sigmoidDerivative(this.preSigActivs[activIndex]);
                    activIndex++;
                }
            }else{
                indexForAllWhenDoingWeightsAndError=0;
                for(int layer=1;layer<layerIndex+1;layer++){//calcs first index in all of next layer
                    indexForAllWhenDoingWeightsAndError=indexForAllWhenDoingWeightsAndError+(this.netStructure[layer]*this.netStructure[layer-1]+this.netStructure[layer]);
                }
                arrayIndexForAllWhenDoingWeightsAndError[layerIndex]=indexForAllWhenDoingWeightsAndError;
                indexForActivsError=0;
                for(int layer=0;layer<layerIndex;layer++){//frist index of activ in current layer
                    indexForActivsError=indexForActivsError+this.netStructure[layer];
                }
                arrayForIndexForActivsError[layerIndex]=indexForActivsError;
                for(int neuronInLayer = 0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                    indexWeightErrorChange=indexForAllWhenDoingWeightsAndError+1+neuronInLayer;//first index of weight
                    sumOfErrorAndWeights=0;
                    for(int neuronInLayerAfter=0;neuronInLayerAfter<this.netStructure[layerIndex+1];neuronInLayerAfter++){
                        sumOfErrorAndWeights=sumOfErrorAndWeights+(errorInLayerAbove[neuronInLayerAfter]*this.allWeightsAndBiases[indexWeightErrorChange]);
                        indexWeightErrorChange=indexWeightErrorChange+1+this.netStructure[layerIndex];
                    }
                    errorInLayer[neuronInLayer]=sumOfErrorAndWeights*sigmoidDerivative(this.preSigActivs[indexForActivsError]);
                    indexForActivsError++;
                }
            }
            //errorInLayer has been generated
            totalAllExplored=0;
            for(int layer = this.netStructure.length-1;layer>layerIndex-1;layer=layer-1){
                totalAllExplored=totalAllExplored + (this.netStructure[layer]*this.netStructure[layer-1]+this.netStructure[layer]);
            }
            indexForAllGrad=this.allWeightsAndBiases.length-totalAllExplored;//first all of biases in this layer
            arrayForIndexForAllGrad[layerIndex]=indexForAllGrad;
            totNeuronsExplored=0;
            for(int layer=0;layer<layerIndex-1;layer++){//first index of activs in layer before
                totNeuronsExplored=totNeuronsExplored+this.netStructure[layer];
            }        
            arrayForTotNeuronsExplored[layerIndex]=totNeuronsExplored;
            for(int neuronInLayer=0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                grad[indexForAllGrad]=errorInLayer[neuronInLayer];//sets grad for bias;
                indexForAllGrad++;
                indexForActivsGrad=totNeuronsExplored;
                for(int neuronInLayerBefore=0;neuronInLayerBefore<this.netStructure[layerIndex-1];neuronInLayerBefore++){
                    grad[indexForAllGrad]=errorInLayer[neuronInLayer]*(this.postSigActivs[indexForActivsGrad]);
                    indexForActivsGrad++;
                    indexForAllGrad++;
                }
            }
            errorInLayerAbove=new double[errorInLayer.length];
            for(int i =0;i<errorInLayer.length;i++){
                errorInLayerAbove[i]=errorInLayer[i];
            }
        }
        this.indexForAllWeightAndError=arrayIndexForAllWhenDoingWeightsAndError;
        this.indexForActivError=arrayForIndexForActivsError;
        this.currentAll=arrayForIndexForAllGrad;
        this.totNeuron=arrayForTotNeuronsExplored;
    }

    public double[]loadBestNetWithThisArchitecture(){
        String lineRead;
        double []bestNet=new double[this.allWeightsAndBiases.length];
        double bestAcc=-0.5;
        int indexNo=0;
        int lineNo=0;
        int lineNoStart=0;
        int lineNoEnd=0;
        double currentAcc=0;
        boolean firstInstance=false;
        boolean isBest=true;
        boolean isSameArch=false;
        try{
            FileReader read = new FileReader(this.FILE_NAME);
            BufferedReader buff = new BufferedReader(read);
            Object[]a=buff.lines().toArray();
            String []s = new String[a.length];
            for(int i=0;i<a.length;i++){
                s[i]=String.valueOf(a[i]);
            }
            for(int i=0;i<s.length;i++){
                if(s[i].contains("Network Structure")){
                    isSameArch=false;
                    String str="Network Structure ";
                    for(int index=0;index<this.netStructure.length;index++){
                        str=str+String.valueOf(this.netStructure[index])+"  ";
                    }
                    if(s[i].equals(str)){
                        isSameArch=true;
                    }
                    //System.out.println("contains net struc but not found "+str+" d "+s[i]);
                }
                if(s[i].contains("Best network (with accuracy")&&isSameArch==true){
                    isBest=false;
                    String num="";
                    for(int c=0;c<s[i].length();c++){
                        boolean isNumOrDot=true;
                        try{
                            double f=Double.parseDouble(String.valueOf((s[i].charAt(c))));
                        }catch(NumberFormatException e){
                            if(s[i].charAt(c)!='.'){
                                isNumOrDot=false;
                            }
                        }
                        if(isNumOrDot==true){
                            num =num+s[i].charAt(c);
                        }
                    }
                    currentAcc=Double.parseDouble(num);
                    if(currentAcc>bestAcc&&isSameArch==true){
                        bestAcc=currentAcc;
                        isBest=true;
                    }
                    if(isBest==true&&isSameArch==true){
                        indexNo=0;
                        firstInstance=true;
                        lineNoStart=i+1;
                        lineNoEnd=i+bestNet.length;
                        int bestIn=0;
                        for(int element=lineNoStart;element<=lineNoEnd;element++){
                           // System.out.println("sdad "+s[element
                                   // ]+" elemnt no "+element+" tot "+this.allWeightsAndBiases.length);
                            bestNet[bestIn]=Double.parseDouble(s[element]);
                            bestIn++;
                        }
                    }
                }
            }
        }catch(IOException e){
            System.out.println("errro "+e+" with file "+FILE_NAME);
            e.printStackTrace();
        }
        return bestNet;
    }
    public void preComputeIndices(){
        double madeUpIn[]=new double[this.netStructure[0]];
        this.feedThroughNet(madeUpIn);
        double[]madeUpOut=new double[this.netStructure[this.netStructure.length-1]];
        calcGradientAttemptFirstRunForIndexes(madeUpOut);
    }
     public void calcGradientAttemptFirstRunForIndexes(double[]desiredOutput){
        //using 1/2 (act-des)^2
        double[]errorInLayer= new double[1];
        double[]errorInLayerAbove=new double[1];
        double sumOfErrorAndWeights;
        double grad[]=new double[this.allWeightsAndBiases.length];
        int indexForAllWhenDoingWeightsAndError;
        int indexWeightErrorChange;
        int indexForActivsError;
        int totalAllExplored;
        int indexForAllGrad;
        int totNeuronsExplored;
        int indexForActivsGrad;
        int activIndex;
        int[] arrayIndexForAllWhenDoingWeightsAndError=new int[this.netStructure.length];
        int[]arrayForIndexForActivsError= new int[this.netStructure.length];
        int[]arrayForIndexForAllGrad=new int[this.netStructure.length];
        int []arrayForTotNeuronsExplored=new int[this.netStructure.length];
        for(int layerIndex=this.netStructure.length-1;layerIndex>0;layerIndex--){
            errorInLayer=new double[this.netStructure[layerIndex]];
            if(layerIndex==this.netStructure.length-1){
                activIndex=this.totalNeurons-this.netStructure[this.netStructure.length-1];
                for(int neuronInLayer=0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                    errorInLayer[neuronInLayer]=(this.postSigActivs[activIndex]-desiredOutput[neuronInLayer])*sigmoidDerivative(this.preSigActivs[activIndex]);
                    activIndex++;
                }
            }else{
                indexForAllWhenDoingWeightsAndError=0;
                for(int layer=1;layer<layerIndex+1;layer++){//calcs first index in all of next layer
                    indexForAllWhenDoingWeightsAndError=indexForAllWhenDoingWeightsAndError+(this.netStructure[layer]*this.netStructure[layer-1]+this.netStructure[layer]);
                }
                arrayIndexForAllWhenDoingWeightsAndError[layerIndex]=indexForAllWhenDoingWeightsAndError;
                indexForActivsError=0;
                for(int layer=0;layer<layerIndex;layer++){//frist index of activ in current layer
                    indexForActivsError=indexForActivsError+this.netStructure[layer];
                }
                arrayForIndexForActivsError[layerIndex]=indexForActivsError;
                for(int neuronInLayer = 0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                    indexWeightErrorChange=indexForAllWhenDoingWeightsAndError+1+neuronInLayer;//first index of weight
                    sumOfErrorAndWeights=0;
                    for(int neuronInLayerAfter=0;neuronInLayerAfter<this.netStructure[layerIndex+1];neuronInLayerAfter++){
                        sumOfErrorAndWeights=sumOfErrorAndWeights+(errorInLayerAbove[neuronInLayerAfter]*this.allWeightsAndBiases[indexWeightErrorChange]);
                        indexWeightErrorChange=indexWeightErrorChange+1+this.netStructure[layerIndex];
                    }
                    errorInLayer[neuronInLayer]=sumOfErrorAndWeights*sigmoidDerivative(this.preSigActivs[indexForActivsError]);
                    indexForActivsError++;
                }
            }
            //errorInLayer has been generated
            totalAllExplored=0;
            for(int layer = this.netStructure.length-1;layer>layerIndex-1;layer=layer-1){
                totalAllExplored=totalAllExplored + (this.netStructure[layer]*this.netStructure[layer-1]+this.netStructure[layer]);
            }
            indexForAllGrad=this.allWeightsAndBiases.length-totalAllExplored;//first all of biases in this layer
            arrayForIndexForAllGrad[layerIndex]=indexForAllGrad;
            totNeuronsExplored=0;
            for(int layer=0;layer<layerIndex-1;layer++){//first index of activs in layer before
                totNeuronsExplored=totNeuronsExplored+this.netStructure[layer];
            }        
            arrayForTotNeuronsExplored[layerIndex]=totNeuronsExplored;
            for(int neuronInLayer=0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                grad[indexForAllGrad]=errorInLayer[neuronInLayer];//sets grad for bias;
                indexForAllGrad++;
                indexForActivsGrad=totNeuronsExplored;
                for(int neuronInLayerBefore=0;neuronInLayerBefore<this.netStructure[layerIndex-1];neuronInLayerBefore++){
                    grad[indexForAllGrad]=errorInLayer[neuronInLayer]*(this.postSigActivs[indexForActivsGrad]);
                    indexForActivsGrad++;
                    indexForAllGrad++;
                }
            }
            errorInLayerAbove=new double[errorInLayer.length];
            for(int i =0;i<errorInLayer.length;i++){
                errorInLayerAbove[i]=errorInLayer[i];
            }
        }
    }
     public static double mod(double x){
         if(x<0.0){
             x=x*-1;
         }
         return x;
     }
     public static double tanH(double x){
         return ((Math.pow(Math.E, x)-Math.pow(Math.E,-x))/(Math.pow(Math.E,x)+Math.pow(Math.E,-x)));
     }
     public static double gradTanH(double x){
         return(1.0-(Math.pow(tanH(x),2)));
     }
     public static double reLU(double x){
         if(x>0.0){
             return x;
         }else{
             return 0.0;
         }
     }
     public static double gradRELU(double x){
         if(x>0.0){
             return 1.0;
         }else{
             return 0.0;
        }
     }
     public static double[]softmax(double[]input){
         double sum=0.0;
         double[]ret=new double[input.length];
         double[]exp=new double[input.length];
         for(int i =0;i<input.length;i++){
             exp[i]=Math.exp(input[i]);
             sum =sum+exp[i];
         }
         for(int i =0;i<input.length;i++){
             ret[i]=exp[i]/sum;
         }
         return ret;
     }
     
    public double[]calcGradientAttemptOptimised(double[]desiredOutput){
        //using  (act-des)^2 (sum of all in out layer)
        double[]errorInLayer= new double[1];
        double[]errorInLayerAbove=new double[1];
        double sumOfErrorAndWeights;
        double grad[]=new double[this.allLength];
        int indexForAllWhenDoingWeightsAndError;
        int indexWeightErrorChange;
        int indexForActivsError;
        //int totalAllExplored;
        int indexForAllGrad;
        int changeD;
        //boolean allZero=false;
        int totNeuronsExplored;
        int indexForActivsGrad;
        int activIndex;
        for(int layerIndex=this.netStructureLengthMinusOne;layerIndex>0;layerIndex--){
            errorInLayer=new double[this.netStructure[layerIndex]];
            if(layerIndex==this.netStructureLengthMinusOne){
                activIndex=this.activIn;
                //allZero=false; //eroror is unlikeyl to be zero and hence not computationally worth checking for assume false, and propogate back to next layer
                //all error is extremely unlikely to be zero so not worthwile to check for cos softmax
                for(int neuronInLayer=0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                    errorInLayer[neuronInLayer]= this.postSigActivs[activIndex] - desiredOutput[neuronInLayer];//softmax
                    
                    //errorInLayer[neuronInLayer]=2*(this.postSigActivs[activIndex]-desiredOutput[neuronInLayer])*sigmoidDerivative(this.preSigActivs[activIndex]);
                    //errorInLayer[neuronInLayer]=(-(desiredOutput[neuronInLayer]/this.postSigActivs[activIndex])+((1-desiredOutput[neuronInLayer])/(1-this.postSigActivs[activIndex])))*gr;//sigmoidDerivative(this.preSigActivs[activIndex])/*gradRELU(this.preSigActivs[activIndex])*/;
                    //errorInLayer[neuronInLayer]=((this.postSigActivs[activIndex]-desiredOutput[neuronInLayer])/(mod(this.postSigActivs[activIndex]-desiredOutput[neuronInLayer])))*sigmoidDerivative(this.preSigActivs[activIndex]);
                    activIndex++;
                }
                if(errorInLayer[0]==0.0&&errorInLayer[1]==0.0){
                    return grad;
                }
                
            }else{
                indexForAllWhenDoingWeightsAndError=this.indexForAllWeightAndError[layerIndex];//added in for opt
               /* for(int layer=1;layer<layerIndex+1;layer++){//calcs first index in all of next layer
                    indexForAllWhenDoingWeightsAndError=indexForAllWhenDoingWeightsAndError+(this.netStructure[layer]*this.netStructure[layer-1]+this.netStructure[layer]);
                }*/
                indexForActivsError=this.indexForActivError[layerIndex];
              /*  indexForActivsError=0;
                for(int layer=0;layer<layerIndex;layer++){//frist index of activ in current layer
                    indexForActivsError=indexForActivsError+this.netStructure[layer];
                }*/
                changeD=1+this.netStructure[layerIndex];
                //allZero=true;
                for(int neuronInLayer = 0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                    indexWeightErrorChange=indexForAllWhenDoingWeightsAndError+1+neuronInLayer;//first index of weight
                    sumOfErrorAndWeights=0;
                    for(int neuronInLayerAfter=0;neuronInLayerAfter<this.netStructure[layerIndex+1];neuronInLayerAfter++){
                        sumOfErrorAndWeights=sumOfErrorAndWeights+(errorInLayerAbove[neuronInLayerAfter]*this.allWeightsAndBiases[indexWeightErrorChange]);
                        //indexWeightErrorChange=indexWeightErrorChange+1+this.netStructure[layerIndex];
                        indexWeightErrorChange=indexWeightErrorChange+changeD;
                    }
                    errorInLayer[neuronInLayer]=sumOfErrorAndWeights*sigmoidDerivative(this.preSigActivs[indexForActivsError])/*gradRELU(this.preSigActivs[indexForActivsError])*/;
                    /*if(allZero){
                        if(errorInLayer[neuronInLayer]!=0.0){
                            allZero=false;//all zero is very unlikely
                        }
                    }*/
                    indexForActivsError++;
                }
            }
            //errorInLayer has been generated
            /*totalAllExplored=0;
            for(int layer = this.netStructure.length-1;layer>layerIndex-1;layer=layer-1){
                totalAllExplored=totalAllExplored + (this.netStructure[layer]*this.netStructure[layer-1]+this.netStructure[layer]);
            }
            indexForAllGrad=this.allWeightsAndBiases.length-totalAllExplored;//first all of biases in this layer*/
           /*if(allZero){
               System.out.println("all zeri");
                break;
            }*/
            indexForAllGrad=this.currentAll[layerIndex];
            /*totNeuronsExplored=0;
            for(int layer=0;layer<layerIndex-1;layer++){//first index of activs in layer before
                totNeuronsExplored=totNeuronsExplored+this.netStructure[layer];
            }    */    
            totNeuronsExplored=this.totNeuron[layerIndex];
            for(int neuronInLayer=0;neuronInLayer<this.netStructure[layerIndex];neuronInLayer++){
                grad[indexForAllGrad]=errorInLayer[neuronInLayer];//sets grad for bias;
                //grad[indexForAllGrad]=0;
                indexForAllGrad++;
                indexForActivsGrad=totNeuronsExplored;
                for(int neuronInLayerBefore=0;neuronInLayerBefore<this.netStructure[layerIndex-1];neuronInLayerBefore++){
                    grad[indexForAllGrad]=errorInLayer[neuronInLayer]*(this.postSigActivs[indexForActivsGrad]);
                    indexForActivsGrad++;
                    indexForAllGrad++;
                }
            }
            //if(allZero==true){
                //break;
            //}
            if(layerIndex==1){
                return grad;
            }
            errorInLayerAbove=new double[errorInLayer.length];
            for(int i =0;i<errorInLayer.length;i++){
                errorInLayerAbove[i]=errorInLayer[i];
            }
        }
        return grad;
    }
     public static double[][] shuffleTrainingDataArray(double[][]array){
        Random rnd = new Random();int in;
        double buff;
        for(int y=0;y<array.length;y++){
            in = rnd.nextInt(array.length);
            for(int x=0;x<array[0].length;x++){
                buff=array[y][x];
                array[y][x]=array[in][x];
                array[in][x]=buff;
            }
        }
        return array;
    }
     public void trainNew(double learningRate, int noOfEpochs,int sampleSize,int totalTrainSize,int printAndTestAfterThisManyIterations){
        //preComputeIndices();
        double changeBy[];
        long timeBefore=System.currentTimeMillis();
        double[][]data=new double[0][0];//each across is one full set of data index 0 is label
        //double out[];
        final int epoc=noOfEpochs;
        final int samp=sampleSize;
        final int h=(totalTrainSize/sampleSize);
        final double sampD=Double.parseDouble(String.valueOf(samp));
        final double rate=learningRate;
        final int trainS=totalTrainSize;
        final int printAfter=printAndTestAfterThisManyIterations;
        double[]bestNet = new double[this.allLength];
        double accuracy;
        double mT[]=new double[this.allLength];
        double vT[]=new double[this.allLength];
        double mHatT[]=new double[this.allLength];
        double vHatT[]=new double[this.allLength];
        double prevMT[]=new double[this.allLength];
        double prevVT[]=new double[this.allLength];
        double t =0.0;
        try{
            double l =this.previousVHat[0];
            for(int i =0;i<this.previousMHat.length;i++){
                prevMT[i]=this.previousMHat[i];
                prevVT[i]=this.previousVHat[i];    
            }
            t=this.timeStep;
        }catch(java.lang.NullPointerException e){
        }
        final double betaOne=0.9;
        final double betaTwo=0.999;
        final double epsilon = Math.pow(10,-8);
        

        double gradOfOne[];
        int indexOfTrain=0;
        //double[][]trainingData=AITumourClasification.getTrainingData();
        double[][]trainingData=AITumourClasification.getTrainingData();
        double testData[][]= AITumourClasification.getTestData();
        int indexPrintAndTest=1;
        double start[]=this.testAccuracy(testData);
        double highestAcc=start[0];
        for(int i =0;i<this.allLength;i++){
            bestNet[i]=this.allWeightsAndBiases[i];
        }
        double desiredOut[]= new double[2];
        double[]inputToNet=new double[this.netStructure[0]];
        int indexData=0;
        System.out.println("Accuracy before training "+highestAcc+" %"+" loss of "+start[1]+" check which error func for tho");
        for(int epoch=1;epoch<=epoc;epoch++){

                trainingData = shuffleTrainingDataArray(trainingData);
            indexOfTrain=0;
            for(int iteration=1;iteration<=h;iteration++){
                changeBy = new double[this.allLength];
                indexData=indexOfTrain;
                indexOfTrain=indexOfTrain+samp;
                for (int sampleNo = 0; sampleNo < samp; sampleNo++) {
                    //correctAnswer = data[sampleNo][0];

                        for(int i=0;i<this.netStructure[0];i++){
                            inputToNet[i]=trainingData[indexData][i+2];
                        }
                        

                    this.feedThroughNet(inputToNet);


                        desiredOut[0]=trainingData[indexData][0];desiredOut[1]=trainingData[indexData][1];
                        indexData++;
                    
                    gradOfOne = this.calcGradientAttemptOptimised(desiredOut);
                    for (int i = 0; i < this.allWeightsAndBiases.length; i++) {
                        //changeBy[i] = Double.parseDouble(String.valueOf((changeBy[i] * sampleNo + gradOfOne[i]))) / Double.parseDouble(String.valueOf(sampleNo + 1.0));//saves massives memory but sliggtly slower
                        changeBy[i]=changeBy[i]+gradOfOne[i];
                    }
                }
                t=t+1.0;
                for(int i=0;i<changeBy.length;i++){
                    changeBy[i]=changeBy[i]/sampD;
                    mT[i]=betaOne*prevMT[i]+(1.0-betaOne) * changeBy[i];
                    vT[i]=betaTwo*prevVT[i]+(1.0-betaTwo)*(changeBy[i]*changeBy[i]);
                    mHatT[i]=mT[i]/(1.0-Math.pow(betaOne,t));
                    vHatT[i]=vT[i]/(1.0-Math.pow(betaTwo,t));
                    prevMT[i]=mT[i];
                    prevVT[i]=vT[i];
                    this.allWeightsAndBiases[i]=this.allWeightsAndBiases[i] - (rate*mHatT[i]/(Math.sqrt(vHatT[i])+epsilon));  
                    //this.allWeightsAndBiases[i]=this.allWeightsAndBiases[i]-(changeBy[i]*rate);
                }
                if(indexPrintAndTest==printAfter){
                    double[]accAndLoss=this.testAccuracy(testData);
                    accuracy = accAndLoss[0];
                    if (accuracy >= highestAcc || highestAcc <= 0) {
                        for (int i = 0; i < this.allWeightsAndBiases.length; i++) {
                            bestNet[i] = this.allWeightsAndBiases[i];
                        }
                        highestAcc = accuracy;
                    }
                    System.out.println("Accuracy " + accuracy + "% after " + epoch + " epochs out of " + noOfEpochs + " epochs. Sample size of " + sampleSize + " with a learning rate of " + rate+" loss of "+accAndLoss[1]+ "iteration "+(iteration)+" out of "+h+" best accuracy of "+highestAcc);
                    indexPrintAndTest=0;
                }
                indexPrintAndTest++;
            }
        }
        this.timeStep=t;
        this.previousMHat=new double[this.allLength];
        this.previousVHat=new double[this.allLength];
        for(int i =0;i<prevMT.length;i++){
            this.previousMHat[i]=prevMT[i];
            this.previousVHat[i]=prevVT[i];
        }
        accuracy = this.testAccuracy(testData)[0];
        System.out.println("Accuracy "+accuracy+" with loss of "+this.testAccuracy(testData)[1]);
        long timeAtEnd = System.currentTimeMillis();
        long timeTaken=timeAtEnd-timeBefore;
        try{
            FileWriter writer = new FileWriter(FILE_NAME,true);
            BufferedWriter bufferedWriter = new BufferedWriter(writer);
            bufferedWriter.write("----------------------------");
            bufferedWriter.newLine();
            bufferedWriter.write("Network Structure ");
            for(int i =0;i<this.netStructure.length;i++){
                bufferedWriter.write(String.valueOf(this.netStructure[i])+"  ");
            }
            bufferedWriter.newLine();
            bufferedWriter.write("All weights and biases");
            bufferedWriter.newLine();
            for(int i =0;i<this.allWeightsAndBiases.length;i++){
                bufferedWriter.write(String.valueOf(this.allWeightsAndBiases[i]));
                bufferedWriter.newLine();
            }
            bufferedWriter.write("No of epochs "+noOfEpochs);
            bufferedWriter.newLine();
            bufferedWriter.write("Final accuracy of "+accuracy);
            bufferedWriter.newLine();
            bufferedWriter.write("time taken in milliseconds "+timeTaken);
            bufferedWriter.newLine();
            bufferedWriter.write("Learning rate of "+learningRate);
            bufferedWriter.newLine();
            bufferedWriter.write("Best network (with accuracy "+highestAcc+"% with this sturcture had a configuration of");
            bufferedWriter.newLine();
            for(int i=0;i<bestNet.length;i++){
                bufferedWriter.write(String.valueOf(bestNet[i]));
                bufferedWriter.newLine();
            }
            bufferedWriter.newLine();
            bufferedWriter.write("Using a sample size of "+sampleSize+" for each epoch out of a total of 60000 samples. Note this doesnt neccesarly apply if this is true:  false");
            bufferedWriter.newLine();
            bufferedWriter.write("----------------------------");
            bufferedWriter.close();
            writer.close();
        }catch(IOException e){
            System.out.println("error with file "+FILE_NAME+" error of "+e);
            System.out.println("Error trace");
            e.printStackTrace();
        }
    }

    public static double[][]getTrainingData(){
        return AITumourClasification.getTrainingData();
    }
    public static double[][]getTestData(){
        return AITumourClasification.getTestData();      
    }
    public double[]testAccuracy(double testData[][]){
        int totalCorrect=0;double[]out;
        double[]desiredOut = new double[2];
        double averageLoss=0.0;int greatestIndex;double err;
        double[]netInput=new double[AITumourClasification.WIDTH_OF_DATA-2];
        for(int y=0;y<testData.length;y++){
            desiredOut[0]=testData[y][0];desiredOut[1]=testData[y][1];
            for(int x=2;x<testData[0].length;x++){
                netInput[x-2]=testData[y][x];
            }
            out=this.feedThroughNet(netInput);
            greatestIndex=0;
            if(out[1]>out[0]){
                greatestIndex=1;
            }
            err=0;
            for(int i=0;i<out.length;i++){
                if(desiredOut[i]!=0.0){
                    err = err+(desiredOut[i]*Math.log(out[i]));
                }
            }
            averageLoss=averageLoss+(err*-1.0);
            if(desiredOut[greatestIndex]==1.0){
                totalCorrect=totalCorrect+1;
            }
        }
        double re = (double)totalCorrect/(double)AITumourClasification.NO_OF_TEST_IMAGES;
        re= re*100.0;
        averageLoss=averageLoss/(double)AITumourClasification.NO_OF_TEST_IMAGES;
        double ret[]={re,averageLoss};
        return ret;
        
    }
    public double[]feedThroughNet(double[]input){
        int currentAllIndex=0;
        double []activsThisLayer;
        double []out = new double[this.netStructure[this.netStructureLengthMinusOne]];
        int indexForActivsInThisLayer;
        double valueOfBias;
        double activVal;
        double[]activsPrevLayer=input;
        int indexForPreSigActivs=0;
        int indexForPostSigActivs=0;
        this.preSigActivs=new double[this.totalNeurons];
        this.postSigActivs=new double[this.totalNeurons];
        for(int i=0;i<this.netStructure[0];i++){//first layer
            this.preSigActivs[indexForPreSigActivs]=input[i];
            this.postSigActivs[indexForPostSigActivs]=input[i];
            indexForPostSigActivs++;
            indexForPreSigActivs++;
        }
        for(int layerIndex=1;layerIndex<this.netStructureLength;layerIndex++){
            indexForActivsInThisLayer=0;
            activsThisLayer=new double[this.netStructure[layerIndex]];
            for(int indexOfLayer=0;indexOfLayer<this.netStructure[layerIndex];indexOfLayer++){//neurons in current kayer
                valueOfBias=this.allWeightsAndBiases[currentAllIndex];
                currentAllIndex++;
                activVal=0;
                for(int prevLayerIndex=0;prevLayerIndex<this.netStructure[layerIndex-1];prevLayerIndex++){//neurons in orev layer
                    activVal=activVal + activsPrevLayer[prevLayerIndex] * this.allWeightsAndBiases[currentAllIndex];
                    currentAllIndex++;

                }
                this.preSigActivs[indexForPreSigActivs]=activVal+valueOfBias;
                indexForPreSigActivs++;
                if(layerIndex!=this.netStructureLengthMinusOne){
                    activsThisLayer[indexForActivsInThisLayer]=sigmoid(this.preSigActivs[indexForPreSigActivs-1]);/*activsThisLayer[indexForActivsInThisLayer]=reLU(this.preSigActivs[indexForPreSigActivs-1]);*/
                    indexForActivsInThisLayer++;
                    this.postSigActivs[indexForPostSigActivs]=activsThisLayer[indexForActivsInThisLayer-1];
                    indexForPostSigActivs++;
                }else if(indexOfLayer==this.netStructure[layerIndex]-1){
                    double preSigLayer[]=new double[this.netStructure[layerIndex]];
                    for(int indexVal=0;indexVal<preSigLayer.length;indexVal++){
                       preSigLayer[indexVal]=this.preSigActivs[this.totalNeuronsMinusLastLayer+indexVal];
                    }
                    activsThisLayer=softmax(preSigLayer);
                    for(int i=0;i<activsThisLayer.length;i++){
                        this.postSigActivs[indexForPostSigActivs]=activsThisLayer[i];
                        indexForPostSigActivs++;
                    }
                }
            }
            activsPrevLayer=activsThisLayer;
            if(layerIndex==this.netStructureLengthMinusOne){
                out = activsThisLayer;
            }
        }
        return out;
    }

    public void setAllWeightsAndBiases(double[]allWAndB){
        if(allWAndB.length!=this.allWeightsAndBiases.length){
            System.out.println("Attempted to change weights and biases to an array of different length ");
        }
        for(int i =0;i<allWAndB.length;i++){
            this.allWeightsAndBiases[i]=allWAndB[i];
        }
        //this.allWeightsAndBiases=allWAndB;
    }
    public static double leakyReLU(double x){
        if(x>0.0){
            return x;
        }else{
            return 0.01*x;
        }
    }
    public static double leakyReLUGrad(double x){
       if(x>0.0){
            return 1;
        }else{
            return 0.01;
        }
    }
    public static double sigmoid(double x){
        if(x>0.0){
            return x;
        }else{
            return 0.0;
        } 
        //return reLU(x);
       //return (tanH(x));
        //return( (1.0)/(1.0+Math.pow(Math.E, -x)));
       //return leakyReLU(x);
    }
    public static double sigmoidDerivative(double x){

       if(x>0.0){
            return 1.0;
        }else{
            return 0.0;
        } 
        //return gradRELU(x);
        //return gradTanH(x);
        //return((sigmoid(x))*(1.0-sigmoid(x)));
        //return leakyReLUGrad(x);
    }
    public  double getRandDoubleBetweenOneAndMinusOne(int numberOfNeuronsInLayerBelow,int numberOfNeuronsInLayer){
        return rnd.nextGaussian()*Math.sqrt(2.0/((double)(numberOfNeuronsInLayer+numberOfNeuronsInLayerBelow)));
       // return (rnd.nextGaussian()*Math.sqrt(1.0/(double)numberOfInputsForWeightsInThisLayer)); //xavier init for tanh/sigmoid
       //return ((rnd.nextDouble()-0.5)*2)*((Math.sqrt(6.0))/(Math.sqrt((double)(numberOfNeuronsInLayer+numberOfNeuronsInLayerBelow))));
    }
}
