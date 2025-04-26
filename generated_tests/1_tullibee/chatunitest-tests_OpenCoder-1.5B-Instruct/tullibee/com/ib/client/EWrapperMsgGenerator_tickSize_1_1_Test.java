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

public class EWrapperMsgGenerator_tickSize_1_1_Test {

    @Test
    public void testTickSize() {
        // Arrange
        EWrapperMsgGenerator msgGenerator = Mockito.mock(EWrapperMsgGenerator.class);
        int tickerId = 123;
        int field = 1;
        int size = 5;
        // Act
        String result = msgGenerator.tickSize(tickerId, field, size);
        // Assert
        assertEquals("id=123  field=BID size=5", result);
    }
}
