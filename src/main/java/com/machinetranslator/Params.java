/**
 * 
 */
package com.machinetranslator;

import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;

/**
 * @author Arnaud
 *
 */

public class Params {

	// Parameters for network configuration
	private WeightInit weightInit = WeightInit.XAVIER;
	private IUpdater updater = new Adam(0.001); 
	
	private CacheMode cacheMode = CacheMode.NONE;
	private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
	
	// Parameters for the training phase (hyper parameters)
	private int batchSize = 20;  
	private int nEpochs = 200;
	
	private int FeatureVectorSize = 88;  // This is the size of the one hot vector
	                                     // Length of this one hot vector is 14 (10 digits and "+"," ",beginning of string and end of string, ....)
	private int maxCharForInputs = 20;  // To try out addition for numbers with different number of digits simply change "numDigits"
	private int maxCharForOutputs = 50;
	private int numHiddenNodes = 256; 
	private int seed = 1234;
	
	// Parameters of the CSV file
	private String commaDelimiter = ",";
	// File attributes index
	private int frenchIdx = 0;
	private int englishIdx = 1;
	
	// Data directory. The model file will be saved here.
    private String dataDir = System.getProperty("user.dir")+ "/src/main/resources/data/";
    private String trainDataFile = System.getProperty("user.dir")+ "/src/main/resources/data/trainData.csv";
    private String testDataFile = System.getProperty("user.dir")+ "/src/main/resources/data/testData.csv";
	
	// Getters
	
	public WeightInit getWeightInit() {
 		return weightInit;
 	}
	
	public IUpdater getUpdater() {
		return updater;
	}
	
	public CacheMode getCacheMode() {
	 	return cacheMode;
	}
	 	
	public WorkspaceMode getWorkspaceMode() {
	 	return workspaceMode;
	}
	
	public int getBatchSize() {
		return batchSize;
	}
	
	public int getNEpochs() {
		return nEpochs;
	}
	
	public int getFeatureVectorSize() {
		return FeatureVectorSize;
	}
	
	public int getMaxCharForInputs() {
		return maxCharForInputs;
	}
	
	public int getMaxCharForOutputs() {
		return maxCharForOutputs;
	}
	
	public int getNumHiddenNodes() {
		return numHiddenNodes;
	}
	
	public int getSeed() {
		return seed;
	}
	
	public String getCommaDelimiter() {
		return commaDelimiter;
	}
	
	public int getFrenchIdx() {
		return frenchIdx;
	}
	
	public int getEnglishIdx() {
		return englishIdx;
	}
	
	public String getDataDir() {
		return dataDir;
	}
	
	public String getTrainDataFile() {
		return trainDataFile;
	}
	
	public String getTestDataFile() {
		return testDataFile;
	}
}
