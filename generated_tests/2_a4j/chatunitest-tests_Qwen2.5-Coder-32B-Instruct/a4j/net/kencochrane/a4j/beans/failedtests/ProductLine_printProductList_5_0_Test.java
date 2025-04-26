package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ProductLine_printProductList_5_0_Test {

    @Mock
    private ProductInfo productInfo;

    @InjectMocks
    private ProductLine productLine;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testPrintProductList_ModeAndProductListAvailable() {
        // Arrange
        String expectedMode = "ONLINE";
        String expectedProductList = "Product1, Product2";
        when(productInfo.printProductList()).thenReturn(expectedProductList);
        productLine.setMode(expectedMode);
        // Act
        String result = productLine.printProductList();
        // Assert
        String expectedOutput = "Mode = " + expectedMode + "\n" + expectedProductList + "\n";
        assertEquals(expectedOutput, result);
    }

    @Test
    public void testPrintProductList_ModeNull() {
        // Arrange
        String expectedProductList = "Product1, Product2";
        when(productInfo.printProductList()).thenReturn(expectedProductList);
        productLine.setMode(null);
        // Act
        String result = productLine.printProductList();
        // Assert
        String expectedOutput = "Mode = null\n" + expectedProductList + "\n";
        assertEquals(expectedOutput, result);
    }

    @Test
    public void testPrintProductList_ProductInfoNull() {
        // Arrange
        productLine.setMode("ONLINE");
        productLine.setProductInfo(null);
        // Act
        String result = productLine.printProductList();
        // Assert
        String expectedOutput = "Mode = ONLINE\nnull\n";
        assertEquals(expectedOutput, result);
    }

    @Test
    public void testPrintProductList_ModeAndProductInfoNull() {
        // Arrange
        productLine.setMode(null);
        productLine.setProductInfo(null);
        // Act
        String result = productLine.printProductList();
        // Assert
        String expectedOutput = "Mode = null\nnull\n";
        assertEquals(expectedOutput, result);
    }
}
