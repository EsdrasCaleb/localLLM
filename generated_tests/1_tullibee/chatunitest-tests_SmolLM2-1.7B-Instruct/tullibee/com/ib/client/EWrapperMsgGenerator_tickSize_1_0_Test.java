// Test method
package com.ib.client;

import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_tickSize_1_0_Test {

    // <Buggy Line>: annotation type not applicable to this kind of declaration
    @ExtendWith(MockitoExtension.class)
    @Test
    public void testTickSize() {
        // Arrange
        int tickerId = 1;
        int field = 2;
        int size = 3;
        // <Buggy Line>: cannot find symbol  symbol:   variable msgGenerator  location: class com.ib.client.EWrapperMsgGenerator_tickSize_1_0_Test
        EWrapperMsgGenerator msgGenerator = new EWrapperMsgGenerator();
        String generatedMessage = msgGenerator.tickSize(tickerId, field, size);
        // Assert
        assertNotNull(generatedMessage);
        assertEquals("id=1  " + TickType.getField(field) + "=" + size, generatedMessage);
    }
}
