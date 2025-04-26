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

public class Item_toString_18_4_Test {

    @Test
    public void testToString() {
        // Arrange
        Item item = new Item();
        // Act
        String expectedOutput = "Asin = 1234567890, Name = Item-1234567890, quantity = 1, ListPrice = 100.99, ourPrice = 100.99, exchangeId = 1234567890, Quantity = 1";
        // Assert
        assertEquals(expectedOutput, item.toString());
    }
}
