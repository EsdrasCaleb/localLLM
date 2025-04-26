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

public class FullProduct_printFullProduct_8_0_Test {

    private FullProduct fullProduct;

    private final ByteArrayOutputStream outputStreamCaptor = new ByteArrayOutputStream();

    @BeforeEach
    public void setUp() {
        fullProduct = new FullProduct();
        System.setOut(new PrintStream(outputStreamCaptor));
    }

    @Test
    public void testPrintFullProduct_WithDetailsOnly() {
        ProductDetails details = Mockito.mock(ProductDetails.class);
        Mockito.when(details.toString()).thenReturn("Product Details");
        fullProduct.setDetails(details);
        fullProduct.printFullProduct();
        String expectedOutput = "Product Details\n\n-- Accessories --\n-- Similar Products --\n";
        assertEquals(expectedOutput, outputStreamCaptor.toString().trim());
    }

    @Test
    public void testPrintFullProduct_WithAccessories() {
        ProductDetails details = Mockito.mock(ProductDetails.class);
        Mockito.when(details.toString()).thenReturn("Product Details");
        fullProduct.setDetails(details);
        ArrayList<String> accessories = new ArrayList<>();
        accessories.add("Accessory 1");
        accessories.add("Accessory 2");
        fullProduct.setAccessories(accessories);
        fullProduct.printFullProduct();
        String expectedOutput = "Product Details\n\n-- Accessories --\nAccessory 1\n\nAccessory 2\n\n-- Similar Products --\n";
        assertEquals(expectedOutput, outputStreamCaptor.toString().trim());
    }

    @Test
    public void testPrintFullProduct_WithSimilarItems() {
        ProductDetails details = Mockito.mock(ProductDetails.class);
        Mockito.when(details.toString()).thenReturn("Product Details");
        fullProduct.setDetails(details);
        ArrayList<String> similarItems = new ArrayList<>();
        similarItems.add("Similar Item 1");
        similarItems.add("Similar Item 2");
        fullProduct.setSimilarItems(similarItems);
        fullProduct.printFullProduct();
        String expectedOutput = "Product Details\n\n-- Accessories --\n-- Similar Products --\nSimilar Item 1\n\nSimilar Item 2\n\n";
        assertEquals(expectedOutput, outputStreamCaptor.toString().trim());
    }

    @Test
    public void testPrintFullProduct_WithAccessoriesAndSimilarItems() {
        ProductDetails details = Mockito.mock(ProductDetails.class);
        Mockito.when(details.toString()).thenReturn("Product Details");
        fullProduct.setDetails(details);
        ArrayList<String> accessories = new ArrayList<>();
        accessories.add("Accessory 1");
        fullProduct.setAccessories(accessories);
        ArrayList<String> similarItems = new ArrayList<>();
        similarItems.add("Similar Item 1");
        fullProduct.setSimilarItems(similarItems);
        fullProduct.printFullProduct();
        String expectedOutput = "Product Details\n\n-- Accessories --\nAccessory 1\n\n-- Similar Products --\nSimilar Item 1\n\n";
        assertEquals(expectedOutput, outputStreamCaptor.toString().trim());
    }

    @Test
    public void testPrintFullProduct_WithEmptyAccessoriesAndSimilarItems() {
        ProductDetails details = Mockito.mock(ProductDetails.class);
        Mockito.when(details.toString()).thenReturn("Product Details");
        fullProduct.setDetails(details);
        fullProduct.setAccessories(new ArrayList<>());
        fullProduct.setSimilarItems(new ArrayList<>());
        fullProduct.printFullProduct();
        String expectedOutput = "Product Details\n\n-- Accessories --\n-- Similar Products --\n";
        assertEquals(expectedOutput, outputStreamCaptor.toString().trim());
    }
}
