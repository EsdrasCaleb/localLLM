package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class BlendedSearch_toString_3_0_Test {

    @Test
    void testToString_withNullProductLines() {
        BlendedSearch blendedSearch = new BlendedSearch();
        String expectedOutput = "productLines is null \n";
        assertEquals(expectedOutput, blendedSearch.toString());
    }

    @Test
    void testToString_withEmptyProductLines() {
        BlendedSearch blendedSearch = new BlendedSearch();
        blendedSearch.setProductLine(new ProductLine[0]);
        String expectedOutput = "productLines is null \n";
        assertEquals(expectedOutput, blendedSearch.toString());
    }

    @Test
    void testToString_withNonEmptyProductLines() {
        ProductLine productLine1 = Mockito.mock(ProductLine.class);
        Mockito.when(productLine1.toString()).thenReturn("ProductLine 1");
        ProductLine productLine2 = Mockito.mock(ProductLine.class);
        Mockito.when(productLine2.toString()).thenReturn("ProductLine 2");
        BlendedSearch blendedSearch = new BlendedSearch();
        ArrayList<ProductLine> productLines = new ArrayList<>(Arrays.asList(productLine1, productLine2));
        blendedSearch.setProductLine(productLines.toArray(new ProductLine[0]));
        String expectedOutput = "ProductLine 1\nProductLine 2\n# of productLines = 2\n";
        assertEquals(expectedOutput, blendedSearch.toString());
    }

    // Additional test case for handling potential null ProductLine objects within the list.
    @Test
    void testToString_withNullProductLineInList() {
        ProductLine productLine1 = Mockito.mock(ProductLine.class);
        Mockito.when(productLine1.toString()).thenReturn("ProductLine 1");
        ProductLine productLine2 = null;
        BlendedSearch blendedSearch = new BlendedSearch();
        ArrayList<ProductLine> productLines = new ArrayList<>(Arrays.asList(productLine1, productLine2));
        blendedSearch.setProductLine(productLines.toArray(new ProductLine[0]));
        // Important:  Handles null ProductLine
        String expectedOutput = "ProductLine 1\nnull\n# of productLines = 2\n";
        assertEquals(expectedOutput, blendedSearch.toString());
    }
}
