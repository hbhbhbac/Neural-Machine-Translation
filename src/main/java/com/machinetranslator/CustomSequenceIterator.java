package com.machinetranslator;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

public class CustomSequenceIterator implements MultiDataSetIterator {

	private static final Logger log = LoggerFactory.getLogger(CustomSequenceIterator.class);
    private MultiDataSetPreProcessor preProcessor;
    private final int batchSize;
    private final int totalBatches;
    private List<Item> itemList = new ArrayList<Item>();
    private int offset = 0;
    
    private static Params params = new Params();

    private static int maxCharForInputs = params.getMaxCharForInputs();
    private static int maxCharForOutputs = params.getMaxCharForOutputs();
    public static int SEQ_VECTOR_DIM = params.getFeatureVectorSize();
    public static final Map<String, Integer> oneHotMap = new HashMap<String, Integer>();
    public static final String[] oneHotOrder = new String[SEQ_VECTOR_DIM];

    private boolean toTestSet = false;
    private int currentBatch = 0;
    
    public CustomSequenceIterator(int totalBatches, List<Item> itemList) {
    	
        this.batchSize = params.getBatchSize();
        this.totalBatches = totalBatches;
        this.itemList = itemList;

        oneHotEncoding();
    }
    

    public MultiDataSet getTestData() {
        toTestSet = true;
        MultiDataSet testData = next(itemList.size());
        reset();
        return testData;
    }
    

    @Override
    public MultiDataSet next(int sampleSize) {
    	
        INDArray encoderSeq, decoderSeq, outputSeq;
        int currentCount = 0;
        String englishDataItem = null; 
        String frenchDataItem = null;
        List<INDArray> encoderSeqList = new ArrayList<>();
        List<INDArray> decoderSeqList = new ArrayList<>();
        List<INDArray> outputSeqList = new ArrayList<>();
        
        Item item = new Item();

        while (currentCount < sampleSize) {   //sampleSize here is batchSize
        	item = itemList.get(currentCount + offset);
        	//System.out.print("Current Count: " + currentCount + "  ");
    		//System.out.println("LINE NUM: " + item.getLineNum());
        	englishDataItem = item.getEnglishDataItem();
    		frenchDataItem = item.getFrenchDataItem();
    		
        	String[] encoderInput = prepToString(englishDataItem);
          	encoderSeqList.add(mapToOneHot(encoderInput));
          	
          	String[] decoderInput = prepToString(frenchDataItem, true);
          	if (toTestSet) {
          		int k = 1;
          		while (k < decoderInput.length) {
          			decoderInput[k] = " ";
          			k++;
          		}
          	}
    		decoderSeqList.add(mapToOneHot(decoderInput));
    		
    		String[] decoderOutput = prepToString(frenchDataItem, false);
    		outputSeqList.add(mapToOneHot(decoderOutput));
    		currentCount++;
        }
        
        encoderSeq = Nd4j.vstack(encoderSeqList); 
        decoderSeq = Nd4j.vstack(decoderSeqList);
        outputSeq = Nd4j.vstack(outputSeqList);
        
        INDArray[] inputs = new INDArray[]{encoderSeq, decoderSeq};
        INDArray[] inputMasks = new INDArray[]{Nd4j.ones(sampleSize, maxCharForInputs), Nd4j.ones(sampleSize, maxCharForOutputs + 1 + 1)};
        INDArray[] labels = new INDArray[]{outputSeq};
        INDArray[] labelMasks = new INDArray[]{Nd4j.ones(sampleSize, maxCharForOutputs + 1 + 1)};
        currentBatch++;
        offset+=sampleSize;
        
        return new org.nd4j.linalg.dataset.MultiDataSet(inputs, labels, inputMasks, labelMasks);
    } 
   
    @Override
    public void reset() {
        currentBatch = 0;
        offset = 0;
        toTestSet = false;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public boolean hasNext() {
        return currentBatch < totalBatches;
    }

    @Override
    public MultiDataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return null;
    }
    

    /** Helper method for encoder input.
     *  Given a string, returns a padded and reversed string array which represents the input to the encoder.
     *  The string array is padded to make its length match the maximum number of characters that each input of the encoder should have.
     *  The missing characters that are replaced by "_".
     *  Eg. in = "Do it." and considering that the maximum number of characters is 10, the method will return {"_","_","_","_",".","t","i"," ","o","D"}
     *  @param in the english data item.
     */ 
    public String[] prepToString(String in) {

        String[] encoded = new String[maxCharForInputs];
        
        //Make the padding
        while (in.length() < maxCharForInputs) {
        	in = "_" + in;
        }

        for (int i = 0; i < encoded.length; i++) {
            encoded[(encoded.length - 1) - i] = Character.toString(in.charAt(i));
        }

        return encoded;
    }
    

    /**
     *   Helper method for the decoder input when GGGooo First, or for decoder output when not GGGooo
     *   Given a string, return a string array which represents the decoder input (or output) given GGGooo First (or not GGGooo First)
     *   Eg. out = "Fait le." (which is the label of "Do it.")
     *           if GGGooo First, then method will return  {"GGGooo","F","a","i","t"," ","l","e",".","_"}
     *           if not GGGooo First, then method will return {"F","a","i","t"," ","l","e",".","_","EEEnnnddd"}
     *   @param out the french data item. 
     */
    public String[] prepToString(String out, boolean goFirst) {
        int start, end;
        String[] decoded = new String[maxCharForOutputs + 1 + 1];
        if (goFirst) {
            decoded[0] = "GGGooo";
            start = 1;
            end = decoded.length - 1;
        } else {
            start = 0;
            end = decoded.length - 2;
            decoded[decoded.length - 1] = "EEEnnnddd";
        }

        int maxIndex = start;
        //add in digits
        for (int i = 0; i < out.length(); i++) {
            decoded[start + i] = Character.toString(out.charAt(i));
            maxIndex ++;
        }

        //needed padding
        while (maxIndex <= end) {
            decoded[maxIndex] = "_";
            maxIndex++;
        }
        return decoded;

    }

    /**
     *   This method takes in an array of strings and return a one hot encoded array of size 1 x FeatureVectorSize x timesteps
     *   Each element in the array indicates a time step
     *   @param toEncode the array of strings.
     */
    private static INDArray mapToOneHot(String[] toEncode) {

        INDArray ret = Nd4j.zeros(1, SEQ_VECTOR_DIM, toEncode.length);
        for (int i = 0; i < toEncode.length; i++) {
            ret.putScalar(0, oneHotMap.get(toEncode[i]), i, 1); 
        }

        return ret;
    }

    public  static String mapToString (INDArray encodeSeq, INDArray decodeSeq) {
        return mapToString(encodeSeq,decodeSeq," --> ");
    }
    
    public static String mapToString(INDArray encodeSeq, INDArray decodeSeq, String sep) {
        String ret = "";
        String [] encodeSeqS = oneHotDecode(encodeSeq);
        String [] decodeSeqS = oneHotDecode(decodeSeq);
        for (int i=0; i<encodeSeqS.length;i++) {
            ret += "\t" + encodeSeqS[i] + sep +decodeSeqS[i] + "\n";
        }
        return ret;
    }
    

    /**
     *   Helper method that takes in a one hot encoded INDArray and returns an interpreted array of strings
     *   toInterpret size batchSize * one_hot_vector_size(FeatureVectorSize) * time_steps
     *   @param toInterpret the one hot encoded INDArray.
     */
    public static String[] oneHotDecode(INDArray toInterpret) {

        String[] decodedString = new String[(int)toInterpret.size(0)];
        INDArray oneHotIndices = Nd4j.argMax(toInterpret, 1); //drops a dimension, so now a two dim array of shape batchSize x time_steps
        for (int i = 0; i < oneHotIndices.size(0); i++) {
            int[] currentSlice = oneHotIndices.slice(i).dup().data().asInt(); //each slice is a batch
            decodedString[i] = mapFromOneHot(currentSlice);
        }
        return decodedString;
    }

    private static String mapFromOneHot(int[] toMap) {
        String ret = "";
        for (int i = 0; i < toMap.length; i++) {
            ret += oneHotOrder[toMap[i]];
        }
        //encoder sequence, needs to be reversed
        if (toMap.length > maxCharForOutputs + 1 + 1) {
            return new StringBuilder(ret).reverse().toString();
        }
        return ret;
    }
    

    /**
     * One hot encoding map
    */
    private static void oneHotEncoding() {

    	int k = 0;
    	for (char chUp = 'a'; chUp <= 'z'; chUp++) {
    		oneHotOrder[k] = Character.toString(chUp);
            oneHotMap.put(Character.toString(chUp), k);
            k++;
    	}
    	
    	k = 26;
    	for (char chDown = 'A'; chDown <= 'Z'; chDown++) {
    		oneHotOrder[k] = Character.toString(chDown);
            oneHotMap.put(Character.toString(chDown), k);
            k++;
    	}
    	
        for (int i = 52; i <= 61; i++) {
            oneHotOrder[i] = String.valueOf(i-51);
            oneHotMap.put(String.valueOf(i-51), i);
        }
        oneHotOrder[62] = " ";
        oneHotMap.put(" ", 62);

        oneHotOrder[63] = "GGGooo";
        oneHotMap.put("GGGooo", 63);

        oneHotOrder[64] = "EEEnnnddd";
        oneHotMap.put("EEEnnnddd", 64);
        
        oneHotOrder[65] = ".";
        oneHotMap.put(".", 65);
        
        oneHotOrder[66] = "!";
        oneHotMap.put("!", 66);
        
        oneHotOrder[67] = "?";
        oneHotMap.put("?", 67);
        
        oneHotOrder[68] = "-";
        oneHotMap.put("-", 68);
        
        oneHotOrder[69] = "'";
        oneHotMap.put("'", 69);
        
        oneHotOrder[70] = "ô";
        oneHotMap.put("ô", 70);
        
        oneHotOrder[71] = "é";
        oneHotMap.put("é", 71);
        
        oneHotOrder[72] = "î";
        oneHotMap.put("î", 72);
        
        oneHotOrder[73] = "û";
        oneHotMap.put("û", 73);
        
        oneHotOrder[74] = "à";
        oneHotMap.put("à", 74);
        
        oneHotOrder[75] = "ç";
        oneHotMap.put("ç", 75);
        
        oneHotOrder[76] = "â";
        oneHotMap.put("â", 76);
        
        oneHotOrder[77] = "ê";
        oneHotMap.put("ê", 77);
        
        oneHotOrder[78] = "Ç";
        oneHotMap.put("Ç", 78);
        
        oneHotOrder[79] = "É";
        oneHotMap.put("É", 79);
        
        oneHotOrder[80] = "è";
        oneHotMap.put("è", 80);
        
        oneHotOrder[81] = "À";
        oneHotMap.put("À", 81);
        
        oneHotOrder[82] = "$";
        oneHotMap.put("$", 82);
        
        oneHotOrder[83] = "_";
        oneHotMap.put("_", 83);
        
        oneHotOrder[84] = ":";
        oneHotMap.put(":", 84);
        
        oneHotOrder[85] = "ù";
        oneHotMap.put("ù", 85);
        
        oneHotOrder[86] = "œ";
        oneHotMap.put("œ", 86);
        
        oneHotOrder[87] = "ï";
        oneHotMap.put("ï", 87);
    }
    

    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }
}
