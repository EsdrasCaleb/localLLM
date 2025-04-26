package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ProductLine_printProductList_5_0_Test {

    private ProductLine productLine;

    private ProductInfo productInfoMock;

    @BeforeEach
    public void setUp() {
        productLine = new ProductLine();
        productInfoMock = mock(ProductInfo.class);
        productLine.setProductInfo(productInfoMock);
    }

    @Test
    public void testPrintProductList_withValidModeAndProductInfo() {
        // Arrange
        productLine.setMode("Test Mode");
        when(productInfoMock.printProductList()).thenReturn("Product Info");
        // Act
        String result = productLine.printProductList();
        // Assert
        assertEquals("Mode = Test Mode\nProduct Info\n", result);
    }

    @Test
    public void testPrintProductList_withNullProductInfo() {
        // Arrange
        productLine.setMode("Test Mode");
        productLine.setProductInfo(null);
        // Act
        String result = productLine.printProductList();
        // Assert
        assertEquals("Mode = Test Mode\nnull\n", result);
    }

    @Test
    public void testPrintProductList_withEmptyMode() {
        // Arrange
        productLine.setMode("");
        when(productInfoMock.printProductList()).thenReturn("Product Info");
        // Act
        String result = productLine.printProductList();
        // Assert
        assertEquals("Mode = \nProduct Info\n", result);
    }
}
