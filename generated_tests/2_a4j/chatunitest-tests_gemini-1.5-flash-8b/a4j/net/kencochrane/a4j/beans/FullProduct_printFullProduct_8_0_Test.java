package net.kencochrane.a4j.beans;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class FullProduct_printFullProduct_8_0_Test {

    @Test
    void printFullProduct_noAccessories_noSimilar() {
        ProductDetails details = Mockito.mock(ProductDetails.class);
        ArrayList<String> accessories = new ArrayList<>();
        ArrayList<String> similarItems = new ArrayList<>();
        FullProduct fullProduct = new FullProduct();
        fullProduct.setDetails(details);
        fullProduct.setAccessories(accessories);
        fullProduct.setSimilarItems(similarItems);
        String expectedOutput = details + "\n" + "-- Accessories --\n" + "-- Similar Products --\n";
        String actualOutput = captureSystemOut(() -> fullProduct.printFullProduct());
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void printFullProduct_withAccessories() {
        ProductDetails details = Mockito.mock(ProductDetails.class);
        ArrayList<String> accessories = new ArrayList<>();
        accessories.add("Accessory 1");
        accessories.add("Accessory 2");
        ArrayList<String> similarItems = new ArrayList<>();
        FullProduct fullProduct = new FullProduct();
        fullProduct.setDetails(details);
        fullProduct.setAccessories(accessories);
        fullProduct.setSimilarItems(similarItems);
        String expectedOutput = details + "\n" + "-- Accessories --\n" + "Accessory 1\n" + "Accessory 2\n" + "-- Similar Products --\n";
        String actualOutput = captureSystemOut(() -> fullProduct.printFullProduct());
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void printFullProduct_withSimilarItems() {
        ProductDetails details = Mockito.mock(ProductDetails.class);
        ArrayList<String> accessories = new ArrayList<>();
        ArrayList<String> similarItems = new ArrayList<>();
        similarItems.add("Similar Item 1");
        similarItems.add("Similar Item 2");
        FullProduct fullProduct = new FullProduct();
        fullProduct.setDetails(details);
        fullProduct.setAccessories(accessories);
        fullProduct.setSimilarItems(similarItems);
        String expectedOutput = details + "\n" + "-- Accessories --\n" + "-- Similar Products --\n" + "Similar Item 1\n" + "Similar Item 2\n";
        String actualOutput = captureSystemOut(() -> fullProduct.printFullProduct());
        assertEquals(expectedOutput, actualOutput);
    }

    private String captureSystemOut(Runnable runnable) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream originalOut = System.out;
        System.setOut(new PrintStream(baos));
        runnable.run();
        System.setOut(originalOut);
        return baos.toString();
    }

    // Dummy class for ProductDetails
    static class ProductDetails {

        @Override
        public String toString() {
            return "Product Details";
        }
    }

    // Dummy class for FullProduct
    static class FullProduct {

        private ProductDetails details;

        private ArrayList<String> accessories;

        private ArrayList<String> similarItems;

        public void setDetails(ProductDetails details) {
            this.details = details;
        }

        public void setAccessories(ArrayList<String> accessories) {
            this.accessories = accessories;
        }

        public void setSimilarItems(ArrayList<String> similarItems) {
            this.similarItems = similarItems;
        }

        public void printFullProduct() {
            System.out.println(details);
            System.out.println("-- Accessories --");
            for (String accessory : accessories) {
                System.out.println(accessory);
            }
            System.out.println("-- Similar Products --");
            for (String similarItem : similarItems) {
                System.out.println(similarItem);
            }
        }
    }
}
