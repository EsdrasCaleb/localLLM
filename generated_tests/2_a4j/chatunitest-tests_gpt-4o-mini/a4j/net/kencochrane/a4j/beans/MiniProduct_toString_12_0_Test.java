package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class MiniProduct_toString_12_0_Test {

    @Test
    public void testToString() {
        // Arrange
        MiniProduct product = new MiniProduct();
        product.setAsin("B00001");
        product.setName("Test Product");
        product.setManufacturer("Test Manufacturer");
        product.setPrice("$10.00");
        product.setImageURL("http://example.com/image.jpg");
        // Act
        String result = product.toString();
        // Assert
        String expected = "B00001 \n Test Product \n Test Manufacturer \n $10.00 \n http://example.com/image.jpg";
        assertEquals(expected, result);
    }

    @Test
    public void testToStringWithNullValues() {
        // Arrange
        MiniProduct product = new MiniProduct();
        product.setAsin(null);
        product.setName(null);
        product.setManufacturer(null);
        product.setPrice(null);
        product.setImageURL(null);
        // Act
        String result = product.toString();
        // Assert
        String expected = "null \n null \n null \n null \n null";
        assertEquals(expected, result);
    }
}
