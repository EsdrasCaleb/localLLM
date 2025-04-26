package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class BlendedSearch_printProductList_4_1_Test {

    private BlendedSearch search;

    private ArrayList<ProductLine> productLines;

    @BeforeEach
    public void setUp() {
        search = spy(BlendedSearch.class);
        productLines = mock(ArrayList.class);
        when(search.getProductLinesArrayList()).thenReturn(productLines);
    }

    @Test
    public void testPrintProductList() {
        // Arrange
        ProductLine productLine1 = mock(ProductLine.class);
        ProductLine productLine2 = mock(ProductLine.class);
        when(productLines.get(0)).thenReturn(productLine1);
        when(productLines.get(1)).thenReturn(productLine2);
        // Act
        String result = search.printProductList();
        // Assert
        assertEquals("productLines is null\n# of productLines = 2\n", result);
    }
}
