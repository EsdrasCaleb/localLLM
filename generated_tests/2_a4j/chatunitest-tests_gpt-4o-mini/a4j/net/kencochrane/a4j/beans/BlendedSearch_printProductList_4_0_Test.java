package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class BlendedSearch_printProductList_4_0_Test {

    private BlendedSearch blendedSearch;

    @BeforeEach
    void setUp() {
        blendedSearch = new BlendedSearch();
    }

    @Test
    void testPrintProductList_WithProductLines() {
        // Arrange
        ProductLine productLine1 = Mockito.mock(ProductLine.class);
        ProductLine productLine2 = Mockito.mock(ProductLine.class);
        Mockito.when(productLine1.printProductList()).thenReturn("Product Line 1 Details");
        Mockito.when(productLine2.printProductList()).thenReturn("Product Line 2 Details");
        ArrayList<ProductLine> productLines = new ArrayList<>();
        productLines.add(productLine1);
        productLines.add(productLine2);
        blendedSearch.setProductLine(productLines.toArray(new ProductLine[0]));
        // Act
        String result = blendedSearch.printProductList();
        // Assert
        String expectedOutput = "Product Line 1 Details\n" + "Product Line 2 Details\n" + "# of productLines = 2\n";
        assertEquals(expectedOutput, result);
    }

    @Test
    void testPrintProductList_NoProductLines() {
        // Arrange
        blendedSearch.setProductLine(new ProductLine[0]);
        // Act
        String result = blendedSearch.printProductList();
        // Assert
        String expectedOutput = "# of productLines = 0\n";
        assertEquals(expectedOutput, result);
    }
}
