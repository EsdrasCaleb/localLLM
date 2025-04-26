package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ProductLine_toString_4_1_Test {

    @Test
    public void testToString() {
        // Arrange
        String mode = "TestMode";
        String productInfo = "TestProductInfo";
        ProductInfo mockedProductInfo = Mockito.mock(ProductInfo.class);
        Mockito.when(mockedProductInfo.toString()).thenReturn(productInfo);
        ProductLine productLine = new ProductLine();
        productLine.setMode(mode);
        productLine.setProductInfo(mockedProductInfo);
        // Act
        String result = productLine.toString();
        // Assert
        assertEquals("Mode = TestMode\nTestProductInfo\n", result);
    }
}
