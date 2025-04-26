package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ProductLine_toString_4_0_Test {

    @Test
    public void testToString() {
        // Arrange
        ProductLine productLine = new ProductLine();
        productLine.setMode("Electronics");
        productLine.setProductInfo(new ProductInfo());
        // Act
        String result = productLine.toString();
        // Assert
        assertEquals("Mode = Electronics\nProductInfo = \n", result);
    }
}
