package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_tickString_4_3_Test {

    // Test method
    @Test
    public void testTickString() {
        // Arrange
        int tickerId = 12345;
        int tickType = TickType.CLOSE;
        String value = "100";
        // Act
        String result = EWrapperMsgGenerator.tickString(tickerId, tickType, value);
        // Assert
        assertEquals("id=12345  CLOSE=100", result);
    }
}
