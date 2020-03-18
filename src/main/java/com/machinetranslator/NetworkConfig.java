/**
 * 
 */
package com.machinetranslator;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author Arnaud
 *
 */

public class NetworkConfig {

	private Params params = new Params();
	
	public ComputationGraph getNetworkConfig() {
		
		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(params.getSeed())
				.weightInit(params.getWeightInit())
				.updater(params.getUpdater())
				.cacheMode(params.getCacheMode())
                .trainingWorkspaceMode(params.getWorkspaceMode())
                .inferenceWorkspaceMode(params.getWorkspaceMode())
				.graphBuilder()
				.setInputTypes(InputType.recurrent(params.getFeatureVectorSize()), InputType.recurrent(params.getFeatureVectorSize()))
				
				//The 2 inputs of the network 
                .addInputs("englishIn", "frenchOut")
                       
                //All the inputs to the encoder will have size = batchSize * FeatureVectorSize * timesteps 
                .addLayer("encoder", new LSTM.Builder()
                		.nIn(params.getFeatureVectorSize())
                		.nOut(params.getNumHiddenNodes())
                		.activation(Activation.SOFTSIGN)
                		.build(), "englishIn")
                
                //Vertex indicating the very last time step of the encoder layer
                .addVertex("lastTimeStep", new LastTimeStepVertex("englishIn"), "encoder")
                
                //Create a vertex that allows the duplication of 2d input to a 3d input
                .addVertex("duplicateTimeStep", new DuplicateToTimeSeriesVertex("frenchOut"), "lastTimeStep")
                
                //The inputs to the decoder will have size = size of output of last timestep of encoder (numHiddenNodes) + size of the other input
                .addLayer("decoder", new LSTM.Builder()
                		.nIn(params.getFeatureVectorSize() + params.getNumHiddenNodes())
                		.nOut(params.getNumHiddenNodes())
                		.activation(Activation.SOFTSIGN)
                		.build(), "frenchOut", "duplicateTimeStep")
                
                .addLayer("output", new RnnOutputLayer.Builder()
                		.nIn(params.getNumHiddenNodes())
                		.nOut(params.getFeatureVectorSize())
                		.activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT)
                		.build(), "decoder")
                
                .setOutputs("output")
				.build();
		
		return new ComputationGraph(conf);
	}
}
