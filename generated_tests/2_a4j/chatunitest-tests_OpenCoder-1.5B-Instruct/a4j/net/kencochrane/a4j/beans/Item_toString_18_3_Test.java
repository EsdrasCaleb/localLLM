package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

public class Item_toString_18_3_Test {

    @Test
    public void testToString() throws Exception {
        // Arrange
        Item item = Mockito.mock(Item.class);
        when(item.getAsin()).thenReturn("123456789");
        when(item.getProductName()).thenReturn("Test Product");
        when(item.getQuantity()).thenReturn("1");
        // Act
        String result = item.toString();
        // Assert
        assertEquals("Asin = 123456789\nName = Test Product\nquantity = 1\n", result);
    }
}
