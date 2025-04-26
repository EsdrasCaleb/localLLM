package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class ProductLine_printProductList_5_0_Test {

    @Mock
    private ProductInfo productInfo;

    @InjectMocks
    private ProductLine productLine;

    @BeforeEach
    public void setUp() {
        productLine.setMode("TestMode");
    }

    @Test
    public void testPrintProductList() {
        // Arrange
        when(productInfo.printProductList()).thenReturn("ProductList");
        // Act
        String result = productLine.printProductList();
        // Assert
        assertEquals("Mode = TestMode\nProductList\n", result);
        verify(productInfo, times(1)).printProductList();
    }

    @Test
    public void testPrintProductListWithNullMode() {
        // Arrange
        productLine.setMode(null);
        when(productInfo.printProductList()).thenReturn("ProductList");
        // Act
        String result = productLine.printProductList();
        // Assert
        assertEquals("Mode = null\nProductList\n", result);
        verify(productInfo, times(1)).printProductList();
    }

    @Test
    public void testPrintProductListWithNullProductInfo() {
        // Arrange
        productLine.setProductInfo(null);
        // Act & Assert
        assertThrows(NullPointerException.class, () -> productLine.printProductList());
    }
}
