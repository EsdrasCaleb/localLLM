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

class EWrapperMsgGenerator_tickString_4_0_Test {

    @Test
    public void testTickString() {
        // Arrange
        int tickerId = 123;
        int tickType = 1;
        String value = "100";
        // Act
        String result = EWrapperMsgGenerator.tickString(tickerId, tickType, value);
        // Assert
        assertEquals("id=123  " + TickType.getField(tickType) + "=" + value, result);
    }
}
