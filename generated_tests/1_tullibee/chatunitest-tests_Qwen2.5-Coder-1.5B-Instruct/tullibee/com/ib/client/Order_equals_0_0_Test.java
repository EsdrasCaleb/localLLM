package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Order_equals_0_0_Test {

    @Test
    void testEquals() throws Exception {
        // Arrange
        Order order1 = new Order();
        Order order2 = new Order();
        Order order3 = new Order();
        // Act
        boolean result1 = order1.equals(order2);
        boolean result2 = order1.equals(order3);
        // Assert
        assertTrue(result1);
        assertFalse(result2);
    }
}
