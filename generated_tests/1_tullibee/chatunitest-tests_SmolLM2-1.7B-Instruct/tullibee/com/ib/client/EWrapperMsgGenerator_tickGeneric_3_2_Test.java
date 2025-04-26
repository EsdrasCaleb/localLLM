package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
class EWrapperMsgGenerator_tickGeneric_3_2_Test {

    @InjectMocks
    private EWrapperMsgGenerator wrapper;

    @Mock
    private TickType tickType;

    @Test
    void testTickGeneric() {
        // Arrange
        int tickerId = 1;
        int tickType = TickType.OPEN;
        double value = 10.0;
        // Act
        String result = wrapper.tickGeneric(tickerId, tickType, value);
        // Assert
        assertEquals("id=1  OPEN=10.0", result);
    }

    @Test
    void testTickGenericWithInvalidTickerId() {
        // Arrange
        int tickerId = -1;
        int tickType = TickType.OPEN;
        double value = 10.0;
        // Act and Assert
        assertThrows(IllegalArgumentException.class, () -> wrapper.tickGeneric(tickerId, tickType, value));
    }

    @Test
    void testTickGenericWithInvalidTickType() {
        // Arrange
        int tickerId = 1;
        int tickType = -1;
        double value = 10.0;
        // Act and Assert
        assertThrows(IllegalArgumentException.class, () -> wrapper.tickGeneric(tickerId, tickType, value));
    }

    @Test
    void testTickGenericWithInvalidValue() {
        // Arrange
        int tickerId = 1;
        int tickType = TickType.OPEN;
        double value = -10.0;
        // Act and Assert
        assertThrows(IllegalArgumentException.class, () -> wrapper.tickGeneric(tickerId, tickType, value));
    }
}
