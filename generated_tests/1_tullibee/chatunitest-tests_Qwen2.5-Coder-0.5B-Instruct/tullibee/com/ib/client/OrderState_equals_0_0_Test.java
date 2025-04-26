package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class OrderState_equals_0_0_Test {

    @Test
    public void testEquals() {
        // Arrange
        OrderState orderState1 = new OrderState("Pending", "100.00", "200.00", "500.00", 0.0, 0.0, 0.0, "USD", "No warning");
        OrderState orderState2 = new OrderState("Pending", "100.00", "200.00", "500.00", 0.0, 0.0, 0.0, "USD", "No warning");
        // Act
        boolean result1 = orderState1.equals(orderState2);
        boolean result2 = orderState2.equals(orderState1);
        // Assert
        assertEquals(result1, true);
        assertEquals(result2, true);
    }
}
