package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class BlendedSearch_printProductList_4_0_Test {

    private BlendedSearch blendedSearch = new BlendedSearch();

    @Test
    public void testPrintProductList() {
        // Arrange
        List<ProductLine> productLines = Arrays.asList(new ProductLine(), new ProductLine(), new ProductLine());
        // Act
        String expectedOutput = "productLines is null \n";
        String actualOutput = blendedSearch.printProductList();
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testPrintProductListWithProductLines() {
        // Arrange
        List<ProductLine> productLines = Arrays.asList(new ProductLine(), new ProductLine(), new ProductLine());
        // Act
        String expectedOutput = "productLines is null \n";
        String actualOutput = blendedSearch.printProductList();
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testPrintProductListWithProductLinesAndNull() {
        // Arrange
        List<ProductLine> productLines = Arrays.asList(new ProductLine(), new ProductLine(), new ProductLine());
        // Act
        String expectedOutput = "productLines is null \n";
        String actualOutput = blendedSearch.printProductList();
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }
}
