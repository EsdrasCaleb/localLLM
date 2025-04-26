package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class BlendedSearch_toString_3_0_Test {

    private BlendedSearch blendedSearch;

    @BeforeEach
    public void setUp() {
        blendedSearch = new BlendedSearch();
    }

    @Test
    public void testToStringWithProductLines() {
        // Create mock ProductLine objects
        ProductLine productLine1 = mock(ProductLine.class);
        ProductLine productLine2 = mock(ProductLine.class);
        // Set up mock objects to return specific values when toString() is called
        when(productLine1.toString()).thenReturn("ProductLine1");
        when(productLine2.toString()).thenReturn("ProductLine2");
        // Create an ArrayList of mock ProductLine objects
        ArrayList<ProductLine> productLines = new ArrayList<>();
        productLines.add(productLine1);
        productLines.add(productLine2);
        // Set the productLines ArrayList using reflection
        try {
            blendedSearch.getClass().getDeclaredField("productLines").setAccessible(true);
            blendedSearch.getClass().getDeclaredField("productLines").set(blendedSearch, productLines);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
        // Invoke the toString() method and assert the expected output
        String expectedOutput = "ProductLine1\nProductLine2\n# of productLines = 2\n";
        String actualOutput = blendedSearch.toString();
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testToStringWithoutProductLines() {
        // Set the productLines ArrayList to null using reflection
        try {
            blendedSearch.getClass().getDeclaredField("productLines").setAccessible(true);
            blendedSearch.getClass().getDeclaredField("productLines").set(blendedSearch, null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
        // Invoke the toString() method and assert the expected output
        String expectedOutput = "productLines is null \n";
        String actualOutput = blendedSearch.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
