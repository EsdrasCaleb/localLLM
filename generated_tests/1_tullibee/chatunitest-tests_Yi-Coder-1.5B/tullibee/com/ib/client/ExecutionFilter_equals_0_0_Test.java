package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class ExecutionFilter_equals_0_0_Test {

    // Test class
    @Test
    public void testEqualsObject() {
        // Arrange
        ExecutionFilter l_this = new ExecutionFilter(1, "ACCT1", "12:00:00", "SPY", "STK", "SMART", "BUY");
        ExecutionFilter l_that = new ExecutionFilter(1, "ACCT1", "12:00:00", "SPY", "STK", "SMART", "BUY");
        // Act
        boolean l_result = l_this.equals(l_that);
        // Assert
        assertEquals(true, l_result);
    }
}
