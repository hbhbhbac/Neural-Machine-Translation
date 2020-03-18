/**
 * 
 */
package com.machinetranslator;

/**
 * @author Arnaud
 *
 */

public class Item {

	private String frenchDataItem;
	private String englishDataItem;
	
    public Item() {
		
	}
	
	public Item(String frenchDataItem, String englishDataItem) {
		this.frenchDataItem = frenchDataItem;
		this.englishDataItem = englishDataItem;
	}
	
	// Getters and Setters
	
	public String getFrenchDataItem() {
		return frenchDataItem;
	}
	
	public String getEnglishDataItem() {
		return englishDataItem;
	}
}
