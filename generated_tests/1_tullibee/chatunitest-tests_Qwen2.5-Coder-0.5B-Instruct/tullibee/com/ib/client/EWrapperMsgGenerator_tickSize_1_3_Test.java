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

class EWrapperMsgGenerator_tickSize_1_3_Test {

    @Test
    public void testTickSize() {
        // Arrange
        int tickerId = 123;
        int field = 0;
        int size = 100;
        // Act
        String result = EWrapperMsgGenerator.tickSize(tickerId, field, size);
        // Assert
        assertEquals("id=123  field=0=100", result);
    }
}
