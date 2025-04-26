package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class FullProduct_printFullProduct_8_0_Test {

    @Mock
    private ProductDetails mockDetails;

    private FullProduct fullProduct;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        fullProduct = new FullProduct();
        setField(fullProduct, "details", mockDetails);
    }

    private void setField(Object target, String fieldName, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

    @Test
    public void testPrintFullProductWithAllData() throws Exception {
        ArrayList<String> mockAccessories = new ArrayList<>(Arrays.asList("Accessory1", "Accessory2"));
        ArrayList<String> mockSimilarItems = new ArrayList<>(Arrays.asList("Item1", "Item2"));
        setField(fullProduct, "accessories", mockAccessories);
        setField(fullProduct, "similarItems", mockSimilarItems);
        PrintStream originalOut = System.out;
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outContent));
        fullProduct.printFullProduct();
        System.setOut(originalOut);
        String expectedOutput = mockDetails.toString() + "\n" + "-- Accessories --\n" + "Accessory1\n" + "Accessory2\n" + "-- Similar Products --\n" + "Item1\n" + "Item2\n";
        assertEquals(expectedOutput, outContent.toString());
    }

    @Test
    public void testPrintFullProductNoAccessories() throws Exception {
        ArrayList<String> mockSimilarItems = new ArrayList<>(Arrays.asList("Item1", "Item2"));
        setField(fullProduct, "accessories", new ArrayList<>());
        setField(fullProduct, "similarItems", mockSimilarItems);
        PrintStream originalOut = System.out;
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outContent));
        fullProduct.printFullProduct();
        System.setOut(originalOut);
        String expectedOutput = mockDetails.toString() + "\n" + "-- Accessories --\n" + "-- Similar Products --\n" + "Item1\n" + "Item2\n";
        assertEquals(expectedOutput, outContent.toString());
    }

    @Test
    public void testPrintFullProductNoSimilarItems() throws Exception {
        ArrayList<String> mockAccessories = new ArrayList<>(Arrays.asList("Accessory1", "Accessory2"));
        setField(fullProduct, "accessories", mockAccessories);
        setField(fullProduct, "similarItems", new ArrayList<>());
        PrintStream originalOut = System.out;
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outContent));
        fullProduct.printFullProduct();
        System.setOut(originalOut);
        String expectedOutput = mockDetails.toString() + "\n" + "-- Accessories --\n" + "Accessory1\n" + "Accessory2\n" + "-- Similar Products --\n";
        assertEquals(expectedOutput, outContent.toString());
    }

    @Test
    public void testPrintFullProductNoData() throws Exception {
        setField(fullProduct, "accessories", new ArrayList<>());
        setField(fullProduct, "similarItems", new ArrayList<>());
        PrintStream originalOut = System.out;
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outContent));
        fullProduct.printFullProduct();
        System.setOut(originalOut);
        String expectedOutput = mockDetails.toString() + "\n" + "-- Accessories --\n" + "-- Similar Products --\n";
        assertEquals(expectedOutput, outContent.toString());
    }
}
