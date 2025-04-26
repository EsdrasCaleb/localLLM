// Test method
package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Date;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;

class EWrapperMsgGenerator_tickGeneric_3_0_Test {

    @Test
    public void testTickGeneric() {
        // Arrange
        int tickerId = 1;
        // changed the value to make it match the method argument
        int tickType = 2;
        double value = 100.5;
        String expected = "id=1  Last Trade=100.5";
        // Act
        String result = EWrapperMsgGenerator.tickGeneric(tickerId, tickType, value);
        // Assert
        assertEquals(expected, result);
    }
}
