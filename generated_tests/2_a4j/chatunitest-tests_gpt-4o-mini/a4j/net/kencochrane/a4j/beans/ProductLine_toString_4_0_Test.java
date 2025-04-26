package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ProductLine_toString_4_0_Test {

    private ProductLine productLine;

    private ProductInfo productInfo;

    @BeforeEach
    public void setUp() {
        productLine = new ProductLine();
        productInfo = mock(ProductInfo.class);
    }

    @Test
    public void testToString_WithModeAndProductInfo() {
        // Arrange
        productLine.setMode("TestMode");
        when(productInfo.toString()).thenReturn("ProductInfoDetails");
        productLine.setProductInfo(productInfo);
        // Act
        String result = productLine.toString();
        // Assert
        String expected = "Mode = TestMode\nProductInfoDetails\n";
        assertEquals(expected, result);
    }

    @Test
    public void testToString_WithNullModeAndProductInfo() {
        // Arrange
        productLine.setMode(null);
        when(productInfo.toString()).thenReturn("ProductInfoDetails");
        productLine.setProductInfo(productInfo);
        // Act
        String result = productLine.toString();
        // Assert
        String expected = "Mode = null\nProductInfoDetails\n";
        assertEquals(expected, result);
    }

    @Test
    public void testToString_WithModeAndNullProductInfo() {
        // Arrange
        productLine.setMode("TestMode");
        productLine.setProductInfo(null);
        // Act
        String result = productLine.toString();
        // Assert
        String expected = "Mode = TestMode\nnull\n";
        assertEquals(expected, result);
    }

    @Test
    public void testToString_WithNullModeAndNullProductInfo() {
        // Arrange
        productLine.setMode(null);
        productLine.setProductInfo(null);
        // Act
        String result = productLine.toString();
        // Assert
        String expected = "Mode = null\nnull\n";
        assertEquals(expected, result);
    }
}
