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
public class EWrapperMsgGenerator_tickString_4_1_Test {

    @Mock
    private EWrapperMsgGenerator focal;

    @InjectMocks
    private EWrapperMsgGenerator testObj;

    @Test
    public void testTickString() {
        // Given
        int tickerId = 123;
        int tickType = 1;
        String value = "testValue";
        // When
        String result = focal.tickString(tickerId, tickType, value);
        // Then
        assertEquals("id=123  1=testValue", result);
    }
}
