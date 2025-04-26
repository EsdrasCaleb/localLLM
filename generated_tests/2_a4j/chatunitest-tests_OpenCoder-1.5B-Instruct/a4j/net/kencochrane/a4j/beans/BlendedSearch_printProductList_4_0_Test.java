package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class BlendedSearch_printProductList_4_0_Test {

    BlendedSearch blendedSearch = new BlendedSearch();

    @Test
    public void testPrintProductList() {
        // Arrange
        ProductLine productLine = mock(ProductLine.class);
        when(productLine.printProductList()).thenReturn("ProductLineMocked");
        blendedSearch.setProductLine(new ProductLine[] { productLine });
        // Act
        String result = blendedSearch.printProductList();
        // Assert
        assertEquals("ProductLineMocked\n# of productLines = 1", result);
    }
}
