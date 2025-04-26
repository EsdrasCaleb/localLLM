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

class EWrapperMsgGenerator_tickGeneric_3_0_Test {

    @Test
    void testTickGeneric() {
        // Given
        int tickerId = 123;
        int tickType = 100;
        double value = 123.45;
        // When
        String message = EWrapperMsgGenerator.tickGeneric(tickerId, tickType, value);
        // Then
        assertEquals("id=123  field=100=123.45", message);
    }
}
